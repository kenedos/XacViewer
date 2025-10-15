# XAC Viewer & Batch Renderer

Tools for viewing and rendering Tree of Savior XAC 3D model files with proper texture support.

## Features

- **OpenGL rendering with full UV texture mapping**
- Parse XAC files and extract mesh geometry, materials, and textures
- Batch process multiple XAC files to JPG images
- Proper DDS texture loading and application
- Isometric view rendering (45° angle, 30° elevation)

## Files

- `batch_render_opengl.py` - **OpenGL batch renderer with proper textures (RECOMMENDED)**
- `batch_render.py` - Simple matplotlib renderer (fast but approximate colors only)
- `lib/` - Minimal XAC parser library (extracted from LaimaEditor)
  - `xac_parser.py` - XAC file format parser
  - `debug_stub.py` - Debug console stub
- `input/` - Folder for your XAC files
- `output/` - Rendered images
- `README.md` - Documentation

## Directory Structure

```
XacViewer/
├── batch_render_opengl.py  # OpenGL renderer (recommended)
├── batch_render.py         # Simple renderer (backup)
├── README.md               # Documentation
├── lib/                    # XAC parser library
│   ├── xac_parser.py
│   └── debug_stub.py
├── input/                  # Put your .xac files here
│   ├── char_hi.ipf/
│   └── char_texture.ipf/
└── output/                 # Rendered images (auto-created)
    └── *_iso.jpg
```

## Usage

### Quick Start - Batch Rendering

**Recommended: OpenGL Renderer (proper UV-mapped textures)**

```bash
# Interactive mode
python batch_render_opengl.py
# Then enter paths when prompted

# Command line mode
python batch_render_opengl.py <input_folder> <output_folder> --textures <texture_folder>
```

**Example:**
```bash
python batch_render_opengl.py input/char_hi.ipf output --textures input/char_texture.ipf
```

**Alternative: Simple Renderer (faster but approximate colors)**

```bash
python batch_render.py <input_folder> <output_folder> --textures <texture_folder>
```

This will:
1. Find all `.xac` files in the input folder recursively
2. Render each one from an isometric view (45° horizontal, 30° elevation)
3. Save as `{filename}_iso.jpg` in the output folder
4. Apply UV-mapped textures with proper DDS support

## Requirements

**For OpenGL Renderer (recommended):**
```bash
pip install PyOpenGL PyOpenGL_accelerate glfw imageio pillow numpy pyrr
```

**For Simple Renderer:**
```bash
pip install matplotlib numpy pillow imageio
```

## Output

- JPG images with isometric view (800x600)
- Filename format: `{model_name}_iso.jpg`
- Default output location: `output/` folder
- Full UV-mapped textures applied

## Credits

- XAC parser extracted from LaimaEditor by the Tree of Savior community
- Parser reference: https://github.com/R-Hidayatullah/tos-parser
- Minimal library version created for standalone batch processing

## Notes

- XAC files are 3D model files used in Tree of Savior
- Textures are optional but recommended for better visual quality
- The batch renderer uses average texture colors for face coloring when full texture mapping isn't available
