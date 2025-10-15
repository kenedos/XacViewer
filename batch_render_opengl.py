"""
OpenGL-based Batch XAC Renderer with Full Texture UV Mapping
Uses headless OpenGL rendering for proper textured output
"""
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from lib.xac_parser import extract_renderable_data


class OpenGLXACRenderer:
    """OpenGL-based renderer for XAC files with full texture support"""

    # Default paths
    DEFAULT_INPUT = "input/char_hi.ipf"
    DEFAULT_OUTPUT = "output"
    DEFAULT_TEXTURES = "input/char_texture.ipf"

    def __init__(self, input_folder=None, output_folder=None, texture_folder=None):
        # Use defaults if not provided
        self.input_folder = Path(input_folder or self.DEFAULT_INPUT)
        self.output_folder = Path(output_folder or self.DEFAULT_OUTPUT)

        if texture_folder is None:
            texture_folder = self.DEFAULT_TEXTURES
        self.texture_folder = [Path(texture_folder)] if texture_folder else []

        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Clean up output folder before processing
        self._cleanup_output_folder()

        # Texture cache
        self.texture_cache = {}

        # Build texture index
        if self.texture_folder:
            print("Building texture index...")
            self._build_texture_index()
            print(f"Indexed {len(self.texture_cache)} texture files")

        # Initialize GLFW and OpenGL
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Create hidden window for offscreen rendering
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(800, 600, "Offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

    def _build_texture_index(self):
        """Build index of all texture files"""
        for tex_folder in self.texture_folder:
            for root, dirs, files in os.walk(tex_folder):
                for file in files:
                    if file.lower().endswith(('.dds', '.png', '.jpg', '.tga', '.bmp')):
                        full_path = os.path.join(root, file)
                        self.texture_cache[file.lower()] = full_path

    def _cleanup_output_folder(self):
        """Remove existing JPG files from output folder"""
        if not self.output_folder.exists():
            return

        jpg_files = list(self.output_folder.glob('*.jpg'))
        if jpg_files:
            print(f"Cleaning up {len(jpg_files)} existing JPG files from output folder...")
            for jpg_file in jpg_files:
                try:
                    jpg_file.unlink()
                except Exception as e:
                    print(f"  Warning: Could not delete {jpg_file.name}: {e}")
            print("Output folder cleaned.")

    def find_texture(self, texture_name):
        """Find texture file"""
        if not texture_name:
            return None

        base_name = os.path.basename(texture_name).lower()
        return self.texture_cache.get(base_name)

    def load_texture_gl(self, texture_path):
        """Load texture into OpenGL"""
        try:
            # Load with imageio (handles DDS)
            img_array = imageio.imread(texture_path)

            # Convert to RGB if needed
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                # Keep RGBA
                pass
            elif img_array.shape[2] == 3:
                # Add alpha channel
                alpha = np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=img_array.dtype) * 255
                img_array = np.concatenate([img_array, alpha], axis=2)

            # Flip vertically for OpenGL
            img_array = np.flipud(img_array)

            # Create OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            format_gl = GL_RGBA if img_array.shape[2] == 4 else GL_RGB
            glTexImage2D(GL_TEXTURE_2D, 0, format_gl, img_array.shape[1], img_array.shape[0],
                        0, format_gl, GL_UNSIGNED_BYTE, img_array)

            return texture_id

        except Exception as e:
            print(f"  Warning: Failed to load texture {texture_path}: {e}")
            return None

    def render_xac(self, xac_path, output_path):
        """Render XAC file to image"""
        try:
            print(f"Processing: {xac_path.name}")

            # Parse XAC file
            meshes, skeleton, _ = extract_renderable_data(str(xac_path))

            if not meshes:
                print(f"  Warning: No meshes found")
                return False

            # Setup OpenGL
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Calculate bounding box
            all_verts = []
            for mesh in meshes:
                if not mesh.is_collision:
                    all_verts.append(mesh.vertices)

            if not all_verts:
                print(f"  Warning: No renderable meshes")
                return False

            vertices = np.vstack(all_verts)
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            size = np.linalg.norm(bbox_max - bbox_min)

            # Setup camera (isometric view)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 800/600, 0.1, size * 10)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Isometric view: 225 degree angle (front-right), 30 degree elevation
            import math
            angle_h = math.radians(270)
            angle_v = math.radians(90)
            distance = size * 1.0

            eye_x = center[0] + distance * math.cos(angle_v) * math.cos(angle_h)
            eye_y = center[1] + distance * math.cos(angle_v) * math.sin(angle_h)
            eye_z = center[2] + distance * math.sin(angle_v)

            gluLookAt(eye_x, eye_y, eye_z,      # Eye position (isometric)
                     center[0], center[1], center[2],  # Look at center
                     0, 0, 1)                          # Up vector

            # Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            for mesh in meshes:
                if mesh.is_collision:
                    continue

                for sub in mesh.sub_meshes:
                    # Load texture
                    texture_id = None
                    if sub['texture_name']:
                        tex_path = self.find_texture(sub['texture_name'])
                        if tex_path:
                            texture_id = self.load_texture_gl(tex_path)

                    if texture_id:
                        glEnable(GL_TEXTURE_2D)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                    else:
                        glDisable(GL_TEXTURE_2D)
                        glColor3f(0.7, 0.7, 0.7)

                    # Draw mesh
                    indices = sub['indices']
                    vertices = mesh.vertices
                    uvs = mesh.uvs if mesh.uvs is not None else None

                    glBegin(GL_TRIANGLES)
                    for i in range(0, len(indices), 3):
                        for j in range(3):
                            idx = indices[i + j]
                            if uvs is not None and idx < len(uvs):
                                glTexCoord2f(uvs[idx][0], uvs[idx][1])
                            if idx < len(vertices):
                                v = vertices[idx]
                                glVertex3f(v[0], v[1], v[2])
                    glEnd()

                    if texture_id:
                        glDeleteTextures([texture_id])

            # Read pixels
            pixels = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(600, 800, 3)
            image = np.flipud(image)  # Flip vertically

            # Save as JPG
            img_pil = Image.fromarray(image)
            img_pil.save(output_path, 'JPEG', quality=90)

            print(f"  [OK] Saved: {output_path.name}")
            return True

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_process(self):
        """Process all XAC files"""
        xac_files = list(self.input_folder.rglob('*.xac'))

        if not xac_files:
            print(f"No XAC files found in {self.input_folder}")
            return

        print(f"Found {len(xac_files)} XAC files")
        if self.texture_folder:
            print(f"Texture folder: {self.texture_folder[0]}")
        print()

        success_count = 0
        for idx, xac_file in enumerate(xac_files, 1):
            relative_path = xac_file.relative_to(self.input_folder)
            output_file = self.output_folder / f"{relative_path.stem}_iso.jpg"

            print(f"[{idx}/{len(xac_files)}] ", end='')
            if self.render_xac(xac_file, output_file):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{len(xac_files)} files")
        print(f"{'='*60}")

    def cleanup(self):
        """Cleanup OpenGL resources"""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch render XAC files with OpenGL')
    parser.add_argument('input_folder', nargs='?', default=None,
                       help=f'Folder containing XAC files (default: {OpenGLXACRenderer.DEFAULT_INPUT})')
    parser.add_argument('output_folder', nargs='?', default=None,
                       help=f'Folder to save JPG images (default: {OpenGLXACRenderer.DEFAULT_OUTPUT})')
    parser.add_argument('--textures', '-t', default=None,
                       help=f'Folder containing textures (default: {OpenGLXACRenderer.DEFAULT_TEXTURES})')

    args = parser.parse_args()

    renderer = OpenGLXACRenderer(args.input_folder, args.output_folder, args.textures)
    try:
        renderer.batch_process()
    finally:
        renderer.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("OpenGL XAC Batch Renderer")
        print("="*60)
        print(f"Default paths:")
        print(f"  Input:    {OpenGLXACRenderer.DEFAULT_INPUT}")
        print(f"  Output:   {OpenGLXACRenderer.DEFAULT_OUTPUT}")
        print(f"  Textures: {OpenGLXACRenderer.DEFAULT_TEXTURES}")
        print()

        use_defaults = input("Use default paths? (Y/n): ").strip().lower()

        if use_defaults == 'n':
            input_folder = input("Enter input folder: ").strip('"')
            output_folder = input("Enter output folder: ").strip('"')
            texture_folder = input("Enter texture folder: ").strip('"')
            renderer = OpenGLXACRenderer(input_folder, output_folder, texture_folder)
        else:
            renderer = OpenGLXACRenderer()

        try:
            renderer.batch_process()
        finally:
            renderer.cleanup()
    else:
        main()
