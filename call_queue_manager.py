import os
import redis
import pickle
import importlib

class CallQueueManager:
    """
    Manages and serializes Celery jobs on a per-call_id basis using Redis for cross-process safety.
    This version is corrected to prevent race conditions using a Redis Lua script.
    """
    def __init__(self):
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
        self.redis = redis.Redis.from_url(redis_url)
        self.in_progress_key = 'call_queue:in_progress'  # Redis set
        self.queue_prefix = 'call_queue:'  # Each call_id gets call_queue:{call_id}
        self.tasks_module = importlib.import_module('tasks')

        # --- CORRECTED PART ---
        # Lua script to atomically add a job and dispatch it ONLY if it's the first one.
        # This prevents race conditions.
        lua_script = """
            -- KEYS[1]: the queue key for the specific call_id (e.g., 'call_queue:123')
            -- KEYS[2]: the global 'in_progress' set key
            -- ARGV[1]: the call_id
            -- ARGV[2]: the serialized job data

            -- Add the new job to the queue for this call_id
            redis.call('RPUSH', KEYS[1], ARGV[2])

            -- Check if another job for this call_id is already in progress
            local in_progress = redis.call('SISMEMBER', KEYS[2], ARGV[1])
            if in_progress == 1 then
                -- A job is already running for this call_id, so do nothing else.
                return nil
            end

            -- If we are here, no job is in progress for this call_id.
            -- This process now has the responsibility to dispatch the *next* job.
            -- Add this call_id to the 'in_progress' set to lock it.
            redis.call('SADD', KEYS[2], ARGV[1])

            -- Pop the next job from the queue to be dispatched.
            return redis.call('LPOP', KEYS[1])
        """
        # Register the script with Redis. It returns a SHA hash for efficient future calls.
        self.add_and_dispatch_script = self.redis.script_load(lua_script)
        # --- END OF CORRECTION ---

    def _queue_key(self, call_id):
        return f'{self.queue_prefix}{call_id}'

    def add_job(self, call_id, celery_task, args=(), kwargs=None):
        """
        Atomically adds a job to the queue and dispatches it if no other job for the
        same call_id is currently in progress.
        """
        print(f"[CALL_QUEUE_MANAGER] Queuing job for call_id: {call_id}")
        if kwargs is None:
            kwargs = {}

        # Serialize the job details for storage in Redis.
        job_data = pickle.dumps((celery_task.__name__, args, kwargs))
        queue_key = self._queue_key(call_id)

        # --- CORRECTED PART ---
        # Execute the Lua script atomically.
        # It returns the job_data to be dispatched, or nil if a job is already in progress.
        job_to_dispatch = self.redis.evalsha(
            self.add_and_dispatch_script,
            2,  # Number of keys
            queue_key, self.in_progress_key,
            call_id, job_data
        )

        if job_to_dispatch:
            # The script returned a job, so we need to dispatch it.
            try:
                task_name, task_args, task_kwargs = pickle.loads(job_to_dispatch)
                celery_task_func = getattr(self.tasks_module, task_name)
                print(f"[CALL_QUEUE_MANAGER] Dispatching job for call_id: {call_id}")
                celery_task_func.apply_async(args=task_args, kwargs=task_kwargs)
            except (pickle.UnpicklingError, AttributeError) as e:
                print(f"[CALL_QUEUE_MANAGER] Error dispatching job for call_id: {call_id}. Error: {e}")
                # Critical: If dispatch fails, we must clear the 'in_progress' flag
                # to prevent the queue from being stuck.
                self.job_done(call_id)
        else:
            # The script returned nil, meaning the job was queued but another
            # was already in progress.
            print(f"[CALL_QUEUE_MANAGER] Job queued for call_id: {call_id}, but another is in progress.")
        # --- END OF CORRECTION ---

    def job_done(self, call_id):
        """
        Marks a job as done and atomically dispatches the next job from the queue if available.
        This method was already implemented correctly using WATCH.
        """
        print(f"[CALL_QUEUE_MANAGER] Job done for call_id: {call_id}. Checking for next job.")
        queue_key = self._queue_key(call_id)

        # This loop handles potential WatchErrors, retrying the transaction if needed.
        while True:
            with self.redis.pipeline() as pipe:
                try:
                    # Watch for changes to the keys we are about to operate on.
                    pipe.watch(queue_key, self.in_progress_key)

                    # Start a transaction block.
                    pipe.multi()

                    # Mark the job as no longer in progress.
                    pipe.srem(self.in_progress_key, call_id)

                    # Try to get the next job from the queue.
                    pipe.lpop(queue_key)

                    # Execute the transaction. results will be a list of the outcomes.
                    results = pipe.execute()

                    # The result of LPOP is the second item in the results list.
                    next_job_data = results[1]

                    if next_job_data:
                        # If there was a next job, we need to dispatch it.
                        # We must first mark the call_id as in_progress again.
                        self.redis.sadd(self.in_progress_key, call_id)
                        task_name, args, kwargs = pickle.loads(next_job_data)
                        celery_task = getattr(self.tasks_module, task_name)
                        print(f"[CALL_QUEUE_MANAGER] Dispatching next job for call_id: {call_id}")
                        celery_task.apply_async(args=args, kwargs=kwargs)
                    else:
                        # The queue is empty.
                        print(f"[CALL_QUEUE_MANAGER] No next job to dispatch for call_id: {call_id} (queue empty).")

                    # If the transaction was successful, break the loop.
                    break
                except redis.WatchError:
                    # Another client modified one of the watched keys.
                    # The transaction was aborted. We'll retry the whole process.
                    print(f"[CALL_QUEUE_MANAGER] WatchError in job_done for call_id: {call_id}, retrying...")
                    continue

# Global instance
# call_queue_manager = CallQueueManager()
# Global instance
call_queue_manager = CallQueueManager() 