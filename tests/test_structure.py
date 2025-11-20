"""
Structure validation test for DHC-Jamba Enhanced.

Validates that all modules can be imported and have correct structure.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    try:
        from dhc_jamba.layers import mamba
        print("✓ mamba module imported")
    except ImportError as e:
        print(f"✗ Failed to import mamba: {e}")
        return False
    
    try:
        from dhc_jamba.layers import attention
        print("✓ attention module imported")
    except ImportError as e:
        print(f"✗ Failed to import attention: {e}")
        return False
    
    try:
        from dhc_jamba.layers import moe
        print("✓ moe module imported")
    except ImportError as e:
        print(f"✗ Failed to import moe: {e}")
        return False
    
    try:
        from dhc_jamba.layers import normalization
        print("✓ normalization module imported")
    except ImportError as e:
        print(f"✗ Failed to import normalization: {e}")
        return False
    
    try:
        from dhc_jamba.core import jamba
        print("✓ jamba module imported")
    except ImportError as e:
        print(f"✗ Failed to import jamba: {e}")
        return False
    
    try:
        from dhc_jamba.core import model
        print("✓ model module imported")
    except ImportError as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        from dhc_jamba.adapters import rl_policy_jamba
        print("✓ rl_policy_jamba module imported")
    except ImportError as e:
        print(f"✗ Failed to import rl_policy_jamba: {e}")
        return False
    
    return True


def test_classes():
    """Test that all classes are defined."""
    print("\n=== Testing Class Definitions ===")
    
    try:
        from dhc_jamba.layers.mamba import MambaLayer
        print("✓ MambaLayer class defined")
    except ImportError as e:
        print(f"✗ MambaLayer not found: {e}")
        return False
    
    try:
        from dhc_jamba.layers.attention import MultiHeadAttention, TransformerFFN
        print("✓ MultiHeadAttention class defined")
        print("✓ TransformerFFN class defined")
    except ImportError as e:
        print(f"✗ Attention classes not found: {e}")
        return False
    
    try:
        from dhc_jamba.layers.moe import MoELayer
        print("✓ MoELayer class defined")
    except ImportError as e:
        print(f"✗ MoELayer not found: {e}")
        return False
    
    try:
        from dhc_jamba.layers.normalization import RMSNorm
        print("✓ RMSNorm class defined")
    except ImportError as e:
        print(f"✗ RMSNorm not found: {e}")
        return False
    
    try:
        from dhc_jamba.core.jamba import JambaBlock, JambaModel, JambaConfig
        print("✓ JambaBlock class defined")
        print("✓ JambaModel class defined")
        print("✓ JambaConfig class defined")
    except ImportError as e:
        print(f"✗ Jamba classes not found: {e}")
        return False
    
    try:
        from dhc_jamba.core.model import DHCJambaModel, DHCJambaConfig
        print("✓ DHCJambaModel class defined")
        print("✓ DHCJambaConfig class defined")
    except ImportError as e:
        print(f"✗ Model classes not found: {e}")
        return False
    
    try:
        from dhc_jamba.adapters.rl_policy_jamba import JambaRLPolicy, JambaRLValue, JambaRLActorCritic
        print("✓ JambaRLPolicy class defined")
        print("✓ JambaRLValue class defined")
        print("✓ JambaRLActorCritic class defined")
    except ImportError as e:
        print(f"✗ RL adapter classes not found: {e}")
        return False
    
    return True


def test_package_init():
    """Test package __init__.py exports."""
    print("\n=== Testing Package Exports ===")
    
    try:
        import dhc_jamba
        print(f"✓ Package version: {dhc_jamba.__version__}")
        
        # Check exports
        expected_exports = [
            'DHCJambaModel',
            'DHCJambaConfig',
            'JambaModel',
            'JambaConfig',
            'JambaBlock',
            'JambaRLPolicy',
            'JambaRLValue',
            'JambaRLActorCritic',
        ]
        
        for export in expected_exports:
            if hasattr(dhc_jamba, export):
                print(f"✓ {export} exported")
            else:
                print(f"✗ {export} not exported")
                return False
        
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        return False
    
    return True


def test_file_structure():
    """Test that all expected files exist."""
    print("\n=== Testing File Structure ===")
    
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    expected_files = [
        'dhc_jamba/__init__.py',
        'dhc_jamba/core/__init__.py',
        'dhc_jamba/core/model.py',
        'dhc_jamba/core/jamba.py',
        'dhc_jamba/layers/__init__.py',
        'dhc_jamba/layers/mamba.py',
        'dhc_jamba/layers/attention.py',
        'dhc_jamba/layers/moe.py',
        'dhc_jamba/layers/normalization.py',
        'dhc_jamba/adapters/__init__.py',
        'dhc_jamba/adapters/rl_policy_jamba.py',
        'dhc_jamba/utils/__init__.py',
        'tests/test_structure.py',
        'setup.py',
        'README.md',
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("=" * 60)
    print("DHC-Jamba Enhanced - Structure Validation")
    print("=" * 60)
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Module Imports", test_imports()))
    results.append(("Class Definitions", test_classes()))
    results.append(("Package Exports", test_package_init()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✓ All structure validation tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
