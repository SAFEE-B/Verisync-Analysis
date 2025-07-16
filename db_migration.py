#!/usr/bin/env python3
"""
Database Schema Migration Script
Migrates existing Verisync call data from legacy schema to optimized schema
"""

import sys
import os
from db_config import get_db, migrate_all_calls_to_optimized_schema, analyze_schema_usage

def main():
    """Main migration function"""
    print("=== Verisync Database Schema Migration ===\n")
    
    try:
        # Get database connection
        db = get_db()
        
        # Analyze current schema usage
        print("1. Analyzing current database schema...")
        analysis = analyze_schema_usage(db)
        
        if "error" in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return False
            
        print(f"📊 Database Analysis Results:")
        print(f"   - Total calls: {analysis['total_calls']}")
        print(f"   - Optimized calls: {analysis['optimized_calls']}")
        print(f"   - Legacy calls: {analysis['legacy_calls']}")
        print(f"   - Optimization percentage: {analysis['optimization_percentage']:.1f}%")
        
        if analysis['storage_analysis']:
            storage = analysis['storage_analysis']
            print(f"   - Potential storage savings: {storage['potential_savings_percentage']:.1f}%")
            print(f"   - Avg transcription size: {storage['avg_transcription_size_bytes']:.0f} bytes")
            print(f"   - Avg conversation size: {storage['avg_conversation_size_bytes']:.0f} bytes")
        
        # Check if migration is needed
        if analysis['legacy_calls'] == 0:
            print("\n✅ All calls are already using optimized schema. No migration needed.")
            return True
            
        # Confirm migration
        print(f"\n⚠️  Found {analysis['legacy_calls']} calls that need migration.")
        
        if len(sys.argv) > 1 and sys.argv[1] == '--auto':
            confirm = 'y'
        else:
            confirm = input("Do you want to proceed with migration? (y/N): ").lower().strip()
        
        if confirm not in ['y', 'yes']:
            print("Migration cancelled.")
            return False
            
        # Perform migration
        print("\n2. Starting migration process...")
        migration_result = migrate_all_calls_to_optimized_schema(db)
        
        if "error" in migration_result:
            print(f"❌ Migration failed: {migration_result['error']}")
            return False
            
        # Show results
        print(f"\n✅ Migration completed successfully!")
        print(f"   - Total calls: {migration_result['total_calls']}")
        print(f"   - Migrated: {migration_result['migrated']}")
        print(f"   - Skipped: {migration_result['skipped']}")
        print(f"   - Errors: {migration_result['errors']}")
        
        # Analyze again to confirm
        print("\n3. Verifying migration results...")
        final_analysis = analyze_schema_usage(db)
        
        if "error" not in final_analysis:
            print(f"✅ Verification successful:")
            print(f"   - Optimized calls: {final_analysis['optimized_calls']}")
            print(f"   - Legacy calls: {final_analysis['legacy_calls']}")
            print(f"   - Optimization percentage: {final_analysis['optimization_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 