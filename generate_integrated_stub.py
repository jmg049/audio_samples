#!/usr/bin/env python3
"""
Generate a single integrated stub file that includes submodules as proper module-level attributes.
This approach solves the type checker recognition problem by putting everything in one file.
"""

import sys
import importlib
import inspect
from pathlib import Path

def get_param_type(param_name: str) -> str:
    """Infer parameter type from common audio processing parameter names."""
    type_map = {
        'sample_rate': 'float',
        'frequency': 'float', 
        'freq': 'float',
        'start_freq': 'float',
        'end_freq': 'float',
        'channels': 'int',
        'channel': 'int',
        'volume': 'float',
        'position_sec': 'float',
        'duration_sec': 'float',
        'enable': 'bool',
        'loop_audio': 'bool',
        'audio': 'AudioSamples',  # Fixed: should be AudioSamples type (no quotes for same module)
        'array': 'Any',
        'data': 'Any',
        'chunk': 'AudioSamples',  # Fixed: should be AudioSamples type (no quotes for same module)
        'other': 'AudioSamples',  # Fixed: should be AudioSamples type (no quotes for same module)
        'copy': 'bool',
        'bits_per_sample': 'int',
        'is_float': 'bool',
        'is_signed': 'bool',
        'buffer_size': 'int',
        'sequence': 'int',
        'timestamp': 'int'
    }
    return type_map.get(param_name, 'Any')

def get_return_type(func, func_name: str) -> str:
    """Infer return type from function name patterns."""
    if func_name.startswith('is_'):
        return 'bool'
    elif func_name.startswith('get_') or func_name in ['position', 'volume']:
        return 'float'
    elif func_name == 'position_samples':
        return 'int'  # Samples are usually integers
    elif func_name == 'create_player':
        return 'AudioPlayer'  # Return proper type (no quotes for same module)
    elif func_name in ['from_numpy', 'load_audio', 'copy', 'normalize', 'scale']:
        return 'AudioSamples'  # Return proper type (no quotes for same module)
    elif 'generator' in func_name:
        return 'GeneratorSource'  # Return proper type (no quotes for same module)
    elif func_name in ['play', 'stop', 'pause', 'seek', 'set_volume', 'set_loop']:
        return 'None'  # These don't return values
    return 'Any'

def safe_signature(obj):
    """Safely get signature, return None if not available."""
    try:
        return inspect.signature(obj)
    except (ValueError, TypeError):
        return None

def render_function(func, name: str, indent: str = "", is_method: bool = False) -> str:
    """Render a function signature with proper type hints."""
    sig = safe_signature(func)
    doc = inspect.getdoc(func)
    
    result = []
    
    # Add documentation if available (but keep it concise)
    if doc and len(doc) < 200:  # Only short docs to avoid clutter
        doc_lines = doc.split('\n')[:3]  # Max 3 lines
        for line in doc_lines:
            if line.strip():
                result.append(f"{indent}# {line.strip()}")
    
    if sig:
        params = []
        
        # Add self parameter for methods
        if is_method:
            params.append("self")
            
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue  # Skip self if it's already in the signature
            param_type = get_param_type(param_name)
            if param.default != inspect.Parameter.empty:
                params.append(f"{param_name}: {param_type} = ...")
            else:
                params.append(f"{param_name}: {param_type}")
        
        return_type = get_return_type(func, name)
        params_str = ", ".join(params)
        result.append(f"{indent}def {name}({params_str}) -> {return_type}: ...")
    else:
        # Fallback - assume it's a method if indent suggests it
        if is_method:
            result.append(f"{indent}def {name}(self, *args: Any, **kwargs: Any) -> Any: ...")
        else:
            result.append(f"{indent}def {name}(*args: Any, **kwargs: Any) -> Any: ...")
    
    return "\n".join(result)

def render_class(cls, name: str, indent: str = "") -> str:
    """Render a class with its methods."""
    result = [f"{indent}class {name}:"]
    
    doc = inspect.getdoc(cls)
    if doc:
        # Keep docstrings concise
        doc_lines = doc.split('\n')[:2]  # Max 2 lines
        result.append(f"{indent}    \"\"\"")
        for line in doc_lines:
            if line.strip():
                result.append(f"{indent}    {line.strip()}")
        result.append(f"{indent}    \"\"\"")
    
    result.append(f"{indent}    def __init__(self, *args: Any, **kwargs: Any) -> None: ...")
    
    # Get class methods
    for attr_name in sorted(dir(cls)):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(cls, attr_name)
            if callable(attr):
                # Render as method with self parameter
                method_stub = render_function(attr, attr_name, indent + "    ", is_method=True)
                result.append(method_stub)
            elif not callable(attr):
                # Class attribute
                result.append(f"{indent}    {attr_name}: Any")
        except:
            pass
    
    return "\n".join(result)

def generate_integrated_stub():
    """Generate a single stub file with all submodules integrated."""
    
    try:
        # Import the main module
        import audio_samples
        main_module = audio_samples
        
        # Import submodules
        import audio_samples.streaming
        import audio_samples.playback
        streaming_module = audio_samples.streaming
        playback_module = audio_samples.playback
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure to run 'maturin develop --features python,streaming,playback' first")
        return
    
    lines = [
        "from __future__ import annotations",
        "import enum", 
        "from typing import Any",
        "",
        "# High-performance audio processing library with Python bindings",
        "# Generated stub file with integrated submodules for proper type checking",
        ""
    ]
    
    # Process main module items
    main_functions = []
    main_classes = []
    main_constants = []
    
    for name in sorted(dir(main_module)):
        if name.startswith('_') or name in ['streaming', 'playback']:
            continue
        try:
            obj = getattr(main_module, name)
            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                main_functions.append((name, obj))
            elif inspect.isclass(obj):
                main_classes.append((name, obj))
            else:
                main_constants.append((name, obj))
        except:
            pass
    
    # Render main module functions
    if main_functions:
        lines.append("# Main Module Functions")
        lines.append("# ====================")
        for name, func in main_functions:
            lines.extend(render_function(func, name, "", is_method=False).split('\n'))
            lines.append("")
    
    # Render main module classes
    if main_classes:
        lines.append("# Main Module Classes") 
        lines.append("# ==================")
        for name, cls in main_classes:
            lines.extend(render_class(cls, name).split('\n'))
            lines.append("")
    
    # Render main module constants
    if main_constants:
        lines.append("# Main Module Constants")
        lines.append("# ====================")
        for name, obj in main_constants:
            lines.append(f"{name}: Any")
        lines.append("")
    
    # Now create submodule classes that act as namespaces
    lines.extend([
        "# Submodules",
        "# ==========", 
        "# These are accessible as audio_samples.streaming and audio_samples.playback",
        ""
    ])
    
    # Render streaming submodule
    lines.extend([
        "class streaming:",
        "    \"\"\"Streaming submodule for audio streaming capabilities\"\"\"",
        ""
    ])
    
    for name in sorted(dir(streaming_module)):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(streaming_module, name)
            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                # These are module-level functions, not methods
                lines.extend(render_function(obj, name, "    ", is_method=False).split('\n'))
            elif inspect.isclass(obj):
                lines.extend(render_class(obj, name, "    ").split('\n'))
            else:
                lines.append(f"    {name}: Any")
            lines.append("")
        except:
            pass
    
    # Render playback submodule  
    lines.extend([
        "class playback:",
        "    \"\"\"Playback submodule for audio playback capabilities\"\"\"",
        ""
    ])
    
    for name in sorted(dir(playback_module)):
        if name.startswith('_'):
            continue
        try:
            obj = getattr(playback_module, name)
            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                # These are module-level functions, not methods
                lines.extend(render_function(obj, name, "    ", is_method=False).split('\n'))
            elif inspect.isclass(obj):
                lines.extend(render_class(obj, name, "    ").split('\n'))
            else:
                lines.append(f"    {name}: Any")
            lines.append("")
        except:
            pass
    
    # Write the integrated stub file
    stub_content = "\n".join(lines)
    
    with open("audio_samples.pyi", "w") as f:
        f.write(stub_content)
    
    print("✅ Generated integrated stub file: audio_samples.pyi")
    print("This single file contains all submodule stubs for proper type checking")

if __name__ == "__main__":
    generate_integrated_stub()