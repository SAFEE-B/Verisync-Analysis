#!/usr/bin/env python3
"""
Test script to demonstrate the XML-like script parsing functionality
"""

import sys
import os

# Add the current directory to the Python path to import from app2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the parsing function from app2
from app2 import parse_script_from_text

def test_script_parsing():
    print("=" * 60)
    print("TESTING XML-LIKE SCRIPT PARSING")
    print("=" * 60)
    
    # Test script with different adherence types
    test_script = """
    <strict>Hello, thank you for calling our company</strict>
    <semantic>My name is John, how can I assist you today?</semantic>
    <topic>I understand your concern and I'm here to help you with that</topic>
    <semantic>Can I please get your account number to pull up your account?</semantic>
    <strict>Is there anything else I can help you with today?</strict>
    <semantic>Thank you for calling. Have a great day!</semantic>
    """
    
    print("Input Script:")
    print(test_script)
    print("\n" + "=" * 60)
    print("PARSED CHECKPOINTS:")
    print("=" * 60)
    
    # Parse the script
    checkpoints = parse_script_from_text(test_script)
    
    # Display results
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\nCheckpoint {i}:")
        print(f"  ID: {checkpoint['checkpoint_id']}")
        print(f"  Type: {checkpoint['adherence_type']}")
        print(f"  Mandatory: {checkpoint['is_mandatory']}")
        print(f"  Weight: {checkpoint['weight']}")
        print(f"  Text: {checkpoint['prompt_text']}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: Parsed {len(checkpoints)} checkpoints")
    print("=" * 60)
    
    # Test the full script loading function
    print("\nTesting get_current_call_script() function:")
    print("=" * 60)
    
    from app2 import get_current_call_script
    
    # Test with custom script
    print("\n1. Testing with custom script:")
    custom_checkpoints = get_current_call_script(test_script)
    print(f"   Result: {len(custom_checkpoints)} checkpoints")
    
    # Test with file (if exists)
    print("\n2. Testing with file fallback:")
    file_checkpoints = get_current_call_script()
    print(f"   Result: {len(file_checkpoints)} checkpoints")
    
    # Show weight distribution
    print("\n3. Weight Distribution Analysis:")
    for checkpoint in custom_checkpoints:
        print(f"   {checkpoint['checkpoint_id']}: Weight={checkpoint['weight']}, Mandatory={checkpoint['is_mandatory']}, Type={checkpoint['adherence_type']}")
    print("=" * 60)
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Empty script
    empty_result = parse_script_from_text("")
    print(f"Empty script: {len(empty_result)} checkpoints (should fallback to default)")
    
    # Invalid format
    invalid_result = parse_script_from_text("Just some text without tags")
    print(f"Invalid format: {len(invalid_result)} checkpoints (should fallback to default)")
    
    # Mixed case tags
    mixed_case = "<STRICT>Upper case tag</STRICT><semantic>Lower case tag</semantic>"
    mixed_result = parse_script_from_text(mixed_case)
    print(f"Mixed case tags: {len(mixed_result)} checkpoints")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_script_parsing() 