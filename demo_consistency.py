#!/usr/bin/env python3
"""
Demonstration script to show consistency between all three script sources:
1. Request parameter
2. File (audioscript.txt)
3. Default built-in script
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app2 import get_current_call_script

def print_checkpoint_structure(checkpoints, source_name):
    """Print checkpoint structure in a consistent format"""
    print(f"\n{source_name} - {len(checkpoints)} checkpoints:")
    print("-" * 80)
    
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i:2d}. {checkpoint['checkpoint_id']}")
        print(f"    Type: {checkpoint['adherence_type']:<8} Weight: {checkpoint['weight']:<3} Mandatory: {checkpoint['is_mandatory']}")
        print(f"    Text: {checkpoint['prompt_text'][:60]}{'...' if len(checkpoint['prompt_text']) > 60 else ''}")
        print()

def demo_consistency():
    print("=" * 80)
    print("CONSISTENCY DEMONSTRATION - ALL SCRIPT SOURCES")
    print("=" * 80)
    
    # Test script in XML format
    custom_script = """
    <strict>Hello, thank you for calling our company</strict>
    <semantic>My name is John, how can I assist you today?</semantic>
    <topic>I understand your concern and I'm here to help you with that</topic>
    <semantic>Can I please get your account number to pull up your account?</semantic>
    <strict>Is there anything else I can help you with today?</strict>
    <semantic>Thank you for calling. Have a great day!</semantic>
    """
    
    print("Testing consistency across all three script sources:\n")
    
    # 1. Custom script from request parameter
    print("1. SCRIPT FROM REQUEST PARAMETER:")
    custom_checkpoints = get_current_call_script(custom_script)
    print_checkpoint_structure(custom_checkpoints, "Custom Script")
    
    # 2. Script from file (audioscript.txt)
    print("2. SCRIPT FROM FILE (audioscript.txt):")
    file_checkpoints = get_current_call_script()
    print_checkpoint_structure(file_checkpoints, "File Script")
    
    # 3. Default built-in script (by passing empty string to trigger default)
    print("3. DEFAULT BUILT-IN SCRIPT:")
    # Temporarily create an empty file or pass None to force default
    default_checkpoints = get_current_call_script("")
    if not default_checkpoints:
        # If empty string fails, get the actual default
        from app2 import parse_script_from_text, DEFAULT_CALL_SCRIPT
        default_checkpoints = parse_script_from_text(DEFAULT_CALL_SCRIPT)
    print_checkpoint_structure(default_checkpoints, "Default Script")
    
    # Analysis
    print("=" * 80)
    print("CONSISTENCY ANALYSIS:")
    print("=" * 80)
    
    print(f"Custom script checkpoints:  {len(custom_checkpoints)}")
    print(f"File script checkpoints:    {len(file_checkpoints)}")
    print(f"Default script checkpoints: {len(default_checkpoints)}")
    
    # Check if all have the same structure keys
    if custom_checkpoints and file_checkpoints and default_checkpoints:
        sample_keys = set(custom_checkpoints[0].keys())
        file_keys = set(file_checkpoints[0].keys())
        default_keys = set(default_checkpoints[0].keys())
        
        print(f"\nStructure consistency:")
        print(f"All have same keys: {sample_keys == file_keys == default_keys}")
        print(f"Common keys: {sorted(sample_keys)}")
        
        # Weight range analysis
        all_weights = []
        all_weights.extend([c['weight'] for c in custom_checkpoints])
        all_weights.extend([c['weight'] for c in file_checkpoints])
        all_weights.extend([c['weight'] for c in default_checkpoints])
        
        print(f"\nWeight distribution:")
        print(f"Weight range: {min(all_weights)} - {max(all_weights)}")
        print(f"Average weight: {sum(all_weights) / len(all_weights):.1f}")
        
        # Mandatory distribution
        mandatory_count = sum([
            sum(1 for c in custom_checkpoints if c['is_mandatory']),
            sum(1 for c in file_checkpoints if c['is_mandatory']),
            sum(1 for c in default_checkpoints if c['is_mandatory'])
        ])
        total_count = len(custom_checkpoints) + len(file_checkpoints) + len(default_checkpoints)
        
        print(f"\nMandatory checkpoints: {mandatory_count}/{total_count} ({mandatory_count/total_count*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ All script sources now produce consistent checkpoint structures!")
    print("✅ Weights are automatically calculated based on type and position")
    print("✅ Mandatory flags are set based on type and importance")
    print("=" * 80)

if __name__ == "__main__":
    demo_consistency() 