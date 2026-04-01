"""
OpenFOAM Mesh File Generator — GUI
===================================
Requires: pip install customtkinter
Place this file in the same folder as generate_mesh_files.py and run it.
"""

import os
import sys
import io
import importlib.util
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    print("customtkinter not found. Run:  pip install customtkinter")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Locate generate_mesh_files.py
# ---------------------------------------------------------------------------
def _find_gen_path():
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "generate_mesh_files.py")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_mesh_files.py")

_GEN_PATH = _find_gen_path()
if not os.path.exists(_GEN_PATH):
    messagebox.showerror("Error", f"Could not find generate_mesh_files.py.\nExpected: {_GEN_PATH}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("generate_mesh_files", _GEN_PATH)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BC_TYPES     = ["inlet", "outlet", "noSlip", "slip"]
TURB_MODELS  = ["kOmegaSST", "kEpsilon", "RNGkEpsilon", "realizableKE", "laminar"]
AXES         = ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Helper — section label
# ---------------------------------------------------------------------------
def section_label(parent, text, row):
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=13, weight="bold")
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 3))


# ---------------------------------------------------------------------------
# Boundary row
# ---------------------------------------------------------------------------
class BoundaryRow(ctk.CTkFrame):
    def __init__(self, parent, on_delete, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.stl_entry = ctk.CTkEntry(self, placeholder_text="e.g. inlet.stl", width=130)
        self.stl_entry.pack(side="left", padx=(0, 5))

        self.bc_var = ctk.StringVar(value="noSlip")
        self.bc_menu = ctk.CTkOptionMenu(
            self, values=BC_TYPES, variable=self.bc_var,
            width=100, command=self._on_bc_change
        )
        self.bc_menu.pack(side="left", padx=(0, 5))

        # refinement level
        self.level_entry = ctk.CTkEntry(self, placeholder_text="level", width=55)
        self.level_entry.insert(0, "0")
        self.level_entry.pack(side="left", padx=(0, 5))

        # flowrate — shown only for inlet
        self.flow_label = ctk.CTkLabel(self, text="Q (m³/s):")
        self.flow_entry = ctk.CTkEntry(self, placeholder_text="0.0123", width=90)

        # ks — shown only for noSlip
        self.ks_label = ctk.CTkLabel(self, text="Ks (m):")
        self.ks_entry = ctk.CTkEntry(self, placeholder_text="None=resolve, 0=smooth", width=140)

        self.delete_btn = ctk.CTkButton(
            self, text="✕", width=30, fg_color="#c0392b",
            hover_color="#e74c3c", command=on_delete
        )
        self.delete_btn.pack(side="right", padx=(5, 0))

        self._on_bc_change("noSlip")

    def _on_bc_change(self, value):
        # hide all optional fields first
        self.flow_label.pack_forget()
        self.flow_entry.pack_forget()
        self.ks_label.pack_forget()
        self.ks_entry.pack_forget()

        if value == "inlet":
            self.flow_label.pack(side="left", padx=(0, 3))
            self.flow_entry.pack(side="left", padx=(0, 5))
        elif value == "noSlip":
            self.ks_label.pack(side="left", padx=(0, 3))
            self.ks_entry.pack(side="left", padx=(0, 5))

    def get(self):
        """Return (stl_name, bc_type, value, level, ks) or raise ValueError."""
        stl = self.stl_entry.get().strip()
        bc  = self.bc_var.get()
        if not stl:
            raise ValueError("STL name cannot be empty.")
        if not stl.lower().endswith(".stl"):
            stl += ".stl"

        # level
        try:
            level = int(self.level_entry.get().strip() or "0")
        except ValueError:
            raise ValueError(f"Level for '{stl}' must be an integer.")

        # flowrate
        if bc == "inlet":
            raw = self.flow_entry.get().strip()
            if not raw:
                raise ValueError(f"Flowrate required for inlet '{stl}'.")
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"Flowrate for '{stl}' must be a number.")
        else:
            value = None

        # ks (wall roughness)
        if bc == "noSlip":
            raw_ks = self.ks_entry.get().strip()
            if raw_ks == "" or raw_ks.lower() == "none":
                ks = None       # resolve the wall (low-Re)
            else:
                try:
                    ks = float(raw_ks)
                except ValueError:
                    raise ValueError(f"Ks for '{stl}' must be a number or left blank.")
        else:
            ks = None

        return (stl, bc, value, level, ks)


# ---------------------------------------------------------------------------
# Probe row
# ---------------------------------------------------------------------------
class ProbeRow(ctk.CTkFrame):
    def __init__(self, parent, on_delete, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        for ph in ("x", "y", "z"):
            e = ctk.CTkEntry(self, placeholder_text=ph, width=80)
            e.pack(side="left", padx=(0, 5))
            setattr(self, f"_{ph}", e)

        ctk.CTkButton(
            self, text="✕", width=30, fg_color="#c0392b",
            hover_color="#e74c3c", command=on_delete
        ).pack(side="right", padx=(5, 0))

    def get(self):
        try:
            x = float(self._x.get().strip())
            y = float(self._y.get().strip())
            z = float(self._z.get().strip())
        except ValueError:
            raise ValueError("Probe x, y, z must all be numbers.")
        return (x, y, z)


# ---------------------------------------------------------------------------
# Refinement region row
# ---------------------------------------------------------------------------
class RefinementRow(ctk.CTkFrame):
    def __init__(self, parent, on_delete, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.name_entry = ctk.CTkEntry(self, placeholder_text="name", width=100)
        self.name_entry.pack(side="left", padx=(0, 5))

        self.type_var = ctk.StringVar(value="box")
        self.type_menu = ctk.CTkOptionMenu(
            self, values=["box", "stl"], variable=self.type_var,
            width=70, command=self._on_type_change
        )
        self.type_menu.pack(side="left", padx=(0, 5))

        self.geom_entry = ctk.CTkEntry(self, placeholder_text="upper bound / stl name", width=160)
        self.geom_entry.pack(side="left", padx=(0, 5))

        for ax in ("lx", "ly", "lz"):
            e = ctk.CTkEntry(self, placeholder_text=ax, width=45)
            e.insert(0, "0")
            e.pack(side="left", padx=(0, 3))
            setattr(self, f"_{ax}", e)

        ctk.CTkButton(
            self, text="✕", width=30, fg_color="#c0392b",
            hover_color="#e74c3c", command=on_delete
        ).pack(side="right", padx=(5, 0))

    def _on_type_change(self, value):
        ph = "upper bound on vertical axis" if value == "box" else "stl filename"
        self.geom_entry.configure(placeholder_text=ph)

    def get(self):
        name = self.name_entry.get().strip()
        rtype = self.type_var.get()
        geom  = self.geom_entry.get().strip()
        if not name:
            raise ValueError("Refinement region name cannot be empty.")
        if not geom:
            raise ValueError(f"Geometry required for region '{name}'.")
        try:
            geom_val = float(geom) if rtype == "box" else geom
        except ValueError:
            raise ValueError(f"Upper bound for box region '{name}' must be a number.")
        try:
            lx = int(self._lx.get().strip() or "0")
            ly = int(self._ly.get().strip() or "0")
            lz = int(self._lz.get().strip() or "0")
        except ValueError:
            raise ValueError(f"Levels for region '{name}' must be integers.")
        return (name, rtype, geom_val, (lx, ly, lz))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OpenFOAM Case File Generator")
        self.geometry("820x960")
        self.resizable(True, True)

        self._boundary_rows:    list[BoundaryRow]    = []
        self._probe_rows:       list[ProbeRow]       = []
        self._refinement_rows:  list[RefinementRow]  = []

        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self)
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        scroll.grid_columnconfigure(0, weight=1)

        r = 0

        # ── Case directory ──────────────────────────────────────────────────
        section_label(scroll, "Case Directory", r); r += 1
        dir_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        dir_frame.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        dir_frame.grid_columnconfigure(0, weight=1)
        self.case_dir_var = tk.StringVar(value=os.getcwd())
        ctk.CTkEntry(dir_frame, textvariable=self.case_dir_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(dir_frame, text="Browse…", width=90, command=self._browse_dir).grid(row=0, column=1)

        # ── Basic settings ──────────────────────────────────────────────────
        section_label(scroll, "Basic Settings", r); r += 1
        basic = ctk.CTkFrame(scroll, fg_color="transparent")
        basic.grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        # vertical axis
        ctk.CTkLabel(basic, text="Vertical axis:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.axis_var = ctk.StringVar(value="z")
        for i, ax in enumerate(AXES):
            ctk.CTkRadioButton(basic, text=ax.upper(), variable=self.axis_var, value=ax).grid(row=0, column=i+1, padx=6)

        # water level
        ctk.CTkLabel(basic, text="Water level (m):").grid(row=1, column=0, sticky="w", pady=(8, 0), padx=(0, 8))
        self.water_level_entry = ctk.CTkEntry(basic, placeholder_text="e.g. 1.0", width=120)
        self.water_level_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))

        # location in mesh
        ctk.CTkLabel(basic, text="locationInMesh:").grid(row=2, column=0, sticky="w", pady=(8, 0), padx=(0, 8))
        loc_f = ctk.CTkFrame(basic, fg_color="transparent")
        loc_f.grid(row=2, column=1, columnspan=3, sticky="w", pady=(8, 0))
        self.loc_x = ctk.CTkEntry(loc_f, placeholder_text="x", width=80)
        self.loc_y = ctk.CTkEntry(loc_f, placeholder_text="y", width=80)
        self.loc_z = ctk.CTkEntry(loc_f, placeholder_text="z", width=80)
        for w in (self.loc_x, self.loc_y, self.loc_z):
            w.pack(side="left", padx=(0, 6))

        # n subdomains
        ctk.CTkLabel(basic, text="Subdomains (parallel):").grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(0, 8))
        self.n_sub_entry = ctk.CTkEntry(basic, placeholder_text="e.g. 4", width=80)
        self.n_sub_entry.insert(0, "4")
        self.n_sub_entry.grid(row=3, column=1, sticky="w", pady=(8, 0))

        # ── controlDict ─────────────────────────────────────────────────────
        section_label(scroll, "controlDict", r); r += 1
        ctrl = ctk.CTkFrame(scroll, fg_color="transparent")
        ctrl.grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        ctk.CTkLabel(ctrl, text="writeInterval (s):").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.write_interval_entry = ctk.CTkEntry(ctrl, placeholder_text="5", width=80)
        self.write_interval_entry.insert(0, "5")
        self.write_interval_entry.grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(ctrl, text="purgeWrite:").grid(row=0, column=2, sticky="w", padx=(20, 8))
        self.purge_write_entry = ctk.CTkEntry(ctrl, placeholder_text="0", width=80)
        self.purge_write_entry.insert(0, "0")
        self.purge_write_entry.grid(row=0, column=3, sticky="w")

        self.iso_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl, text="Write alpha.water isoSurface (vtk)", variable=self.iso_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        # ── Turbulence ──────────────────────────────────────────────────────
        section_label(scroll, "Turbulence", r); r += 1
        turb = ctk.CTkFrame(scroll, fg_color="transparent")
        turb.grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        ctk.CTkLabel(turb, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.turb_var = ctk.StringVar(value="kOmegaSST")
        ctk.CTkOptionMenu(turb, values=TURB_MODELS, variable=self.turb_var, width=160).grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(turb, text="U ref (m/s):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.turb_U = ctk.CTkEntry(turb, placeholder_text="1.0", width=90)
        self.turb_U.insert(0, "1.0")
        self.turb_U.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ctk.CTkLabel(turb, text="Intensity I:").grid(row=1, column=2, sticky="w", padx=(20, 8), pady=(8, 0))
        self.turb_I = ctk.CTkEntry(turb, placeholder_text="0.05", width=90)
        self.turb_I.insert(0, "0.05")
        self.turb_I.grid(row=1, column=3, sticky="w", pady=(8, 0))

        ctk.CTkLabel(turb, text="Length scale L (m):").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.turb_L = ctk.CTkEntry(turb, placeholder_text="0.1", width=90)
        self.turb_L.insert(0, "0.1")
        self.turb_L.grid(row=2, column=1, sticky="w", pady=(8, 0))

        # ── Boundaries ──────────────────────────────────────────────────────
        section_label(scroll, "Boundaries", r); r += 1

        # column headers
        hdr = ctk.CTkFrame(scroll, fg_color="transparent")
        hdr.grid(row=r, column=0, sticky="w", pady=(0, 2)); r += 1
        for txt, w in [("STL filename", 130), ("BC type", 100), ("Level", 55), ("Flowrate / Ks", 160)]:
            ctk.CTkLabel(hdr, text=txt, width=w, anchor="w").pack(side="left", padx=(0, 5))

        self.boundary_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.boundary_container.grid(row=r, column=0, sticky="ew", pady=(0, 4)); r += 1
        ctk.CTkButton(scroll, text="+ Add Boundary", width=140, command=self._add_boundary).grid(
            row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # seed defaults
        defaults = [
            ("inlet.stl",  "inlet",  "0.0123", "0", ""),
            ("bottom.stl", "noSlip", "",        "0", "0.0"),
            ("outlet.stl", "outlet", "",        "0", ""),
            ("rock.stl",   "noSlip", "",        "0", "0.005"),
            ("side.stl",   "slip",   "",        "0", ""),
            ("top.stl",    "slip",   "",        "0", ""),
        ]
        for stl, bc, flow, level, ks in defaults:
            self._add_boundary(stl=stl, bc=bc, flow=flow, level=level, ks=ks)

        # ── Refinement regions ───────────────────────────────────────────────
        section_label(scroll, "Refinement Regions  (name | type | geom | lx ly lz)", r); r += 1
        self.refinement_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.refinement_container.grid(row=r, column=0, sticky="ew", pady=(0, 4)); r += 1
        ctk.CTkButton(scroll, text="+ Add Region", width=140, command=self._add_refinement).grid(
            row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # ── Probes ───────────────────────────────────────────────────────────
        section_label(scroll, "Probes  (x  y  z)", r); r += 1
        self.probe_container = ctk.CTkFrame(scroll, fg_color="transparent")
        self.probe_container.grid(row=r, column=0, sticky="ew", pady=(0, 4)); r += 1
        ctk.CTkButton(scroll, text="+ Add Probe", width=140, command=self._add_probe).grid(
            row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # ── Log ──────────────────────────────────────────────────────────────
        section_label(scroll, "Output Log", r); r += 1
        self.log_box = ctk.CTkTextbox(scroll, height=180, state="disabled")
        self.log_box.grid(row=r, column=0, sticky="ew", pady=(0, 10)); r += 1

        # ── Generate ─────────────────────────────────────────────────────────
        ctk.CTkButton(
            scroll, text="Generate Files", height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._generate
        ).grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1

    # -----------------------------------------------------------------------
    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select OpenFOAM Case Directory")
        if d:
            self.case_dir_var.set(d)

    def _add_boundary(self, stl="", bc="noSlip", flow="", level="0", ks=""):
        def remove(row):
            if row in self._boundary_rows:
                self._boundary_rows.remove(row)
            row.destroy()
        w = BoundaryRow(self.boundary_container, on_delete=lambda r=None: None)
        w.pack(fill="x", pady=2)
        # wire delete after creation so we can reference w
        w.delete_btn.configure(command=lambda: remove(w))
        if stl:   w.stl_entry.insert(0, stl)
        if bc in BC_TYPES:
            w.bc_var.set(bc); w._on_bc_change(bc)
        if flow:  w.flow_entry.insert(0, flow)
        if level: w.level_entry.delete(0, "end"); w.level_entry.insert(0, level)
        if ks:    w.ks_entry.insert(0, ks)
        self._boundary_rows.append(w)

    def _add_probe(self):
        def remove(row):
            if row in self._probe_rows: self._probe_rows.remove(row)
            row.destroy()
        w = ProbeRow(self.probe_container, on_delete=lambda: None)
        w.pack(fill="x", pady=2)
        w.pack_slaves()[-1].configure(command=lambda: remove(w))  # rewire delete
        self._probe_rows.append(w)

    def _add_refinement(self):
        def remove(row):
            if row in self._refinement_rows: self._refinement_rows.remove(row)
            row.destroy()
        w = RefinementRow(self.refinement_container, on_delete=lambda: None)
        w.pack(fill="x", pady=2)
        # find delete button and rewire
        for child in w.winfo_children():
            if isinstance(child, ctk.CTkButton) and child.cget("text") == "✕":
                child.configure(command=lambda: remove(w))
        self._refinement_rows.append(w)

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # -----------------------------------------------------------------------
    def _generate(self):
        # clear log
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        # --- validate -------------------------------------------------------
        case_dir = self.case_dir_var.get().strip()
        if not os.path.isdir(case_dir):
            messagebox.showerror("Error", f"Case directory does not exist:\n{case_dir}")
            return

        try:
            water_level = float(self.water_level_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Water level must be a number."); return

        try:
            lx = float(self.loc_x.get()); ly = float(self.loc_y.get()); lz = float(self.loc_z.get())
        except ValueError:
            messagebox.showerror("Error", "locationInMesh x, y, z must be numbers."); return
        locationInMesh = (lx, ly, lz)

        try:
            write_interval = float(self.write_interval_entry.get().strip())
            purge_write    = int(self.purge_write_entry.get().strip())
            n_subdomains   = int(self.n_sub_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "writeInterval, purgeWrite and subdomains must be numbers."); return

        try:
            turb_U = float(self.turb_U.get().strip())
            turb_I = float(self.turb_I.get().strip())
            turb_L = float(self.turb_L.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Turbulence U, I and L must be numbers."); return

        vertical_axis    = self.axis_var.get()
        turbulence_model = self.turb_var.get()
        write_iso        = self.iso_var.get()

        boundaries = []
        for row in self._boundary_rows:
            try:
                boundaries.append(row.get())
            except ValueError as e:
                messagebox.showerror("Boundary Error", str(e)); return
        if not boundaries:
            messagebox.showerror("Error", "Please add at least one boundary."); return

        probes = []
        for row in self._probe_rows:
            try:
                probes.append(row.get())
            except ValueError as e:
                messagebox.showerror("Probe Error", str(e)); return

        refinement_regions = []
        for row in self._refinement_rows:
            try:
                refinement_regions.append(row.get())
            except ValueError as e:
                messagebox.showerror("Refinement Error", str(e)); return

        stl_files = [b[0] for b in boundaries]

        system_dir   = os.path.join(case_dir, "system")
        zero_dir     = os.path.join(case_dir, "0")
        constant_dir = os.path.join(case_dir, "constant")

        # --- run generators -------------------------------------------------
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # also redirect cwd so STL scanning works
        old_cwd = os.getcwd()
        os.chdir(case_dir)

        try:
            gen.write_block_mesh_dict(system_dir)
            gen.write_snappy_hex_mesh_dict(stl_files, locationInMesh, boundaries, refinement_regions, vertical_axis, system_dir)
            gen.write_surface_feature_extract_dict(stl_files, system_dir)
            gen.write_U(boundaries, zero_dir)
            gen.write_alpha_water(boundaries, zero_dir)
            gen.write_p_rgh(boundaries, zero_dir)
            gen.write_g(vertical_axis, constant_dir)
            gen.write_hRef(water_level, constant_dir)
            gen.write_control_dict(write_interval, purge_write, probes, write_iso, system_dir)
            gen.write_set_fields_dict(water_level, vertical_axis, system_dir)
            gen.write_fv_solution(system_dir)
            gen.write_fv_schemes(system_dir)
            gen.write_turbulence_properties(turbulence_model, constant_dir)
            gen.write_transport_properties(constant_dir)
            gen.write_decompose_par_dict(n_subdomains, system_dir)

            if turbulence_model in ("kEpsilon", "RNGkEpsilon", "realizableKE"):
                gen.write_k_kepsilon(boundaries, turb_U, turb_I, zero_dir)
                gen.write_nut_kepsilon(boundaries, zero_dir)
                gen.write_epsilon(boundaries, turb_U, turb_I, turb_L, zero_dir)
            elif turbulence_model == "kOmegaSST":
                gen.write_k_komegasst(boundaries, turb_U, turb_I, zero_dir)
                gen.write_nut_komegasst(boundaries, zero_dir)
                gen.write_omega(boundaries, turb_U, turb_I, turb_L, zero_dir)

            success = True
        except Exception as e:
            success = False
            error_msg = str(e)
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)

        for line in buffer.getvalue().splitlines():
            self._log(line)

        if success:
            self._log("\n✅ All files generated successfully.")
            messagebox.showinfo("Done", "All files generated successfully!")
        else:
            self._log(f"\n❌ Error: {error_msg}")
            messagebox.showerror("Generation Error", error_msg)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
