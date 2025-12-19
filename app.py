# =========================================================
# Image Processing Studio – TEAM ELMOLOK  GOLD EDITION
# PREMIUM GUI VERSION (FINAL FIXED & UPDATED)
# =========================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk 
import numpy as np
import random
import math
import code as ops  # Make sure this file exists

# ===================== Colors & Assets =====================
COLORS = {
    "bg": "#f1f5f9",          # Light Blue-Grey Background
    "card_bg": "#ffffff",     # Pure White for Cards
    "primary": "#4f46e5",     # Indigo Blue (Modern)
    "primary_hover": "#4338ca",
    "text_main": "#1e293b",   # Dark Slate
    "text_sub": "#64748b",    # Muted Slate
    "splash_bg": "#0f172a",   # Midnight Blue
    "gold": "#fbbf24",        # Amber Gold
    "gold_dark": "#b45309",   # Darker Gold
    "accent": "#ef4444"       # Red for subtract/danger
}

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUB = ("Segoe UI", 12)
FONT_BTN = ("Segoe UI", 10, "bold")

# ===================== Splash Screen (Ultra Premium Fixed) =====================
class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.0)  # Start invisible for fade-in

        # --- Dimensions & Centering ---
        w, h = 700, 450
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))

        # --- Colors ---
        self.c_bg = COLORS["splash_bg"]
        self.c_gold = COLORS["gold"]
        self.c_gold_dark = COLORS["gold_dark"]
        self.c_text_sub = "#94a3b8"

        # --- Main Canvas (The Engine) ---
        self.canvas = tk.Canvas(self, bg=self.c_bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Assets Init ---
        self.particles = []
        self.loading_progress = 0
        self.pulse_direction = 1
        self.pulse_size = 0
        
        # --- Build Scene ---
        self.create_particles(w, h)
        self.draw_logo(w / 2, h / 2 - 50)
        self.draw_text(w / 2, h / 2 + 40)
        self.draw_progress_bar(w / 2, h - 60, w - 100)

        # --- Start Animations ---
        self.fade_in_window()
        self.animate_loop()

    def create_particles(self, w, h):
        for _ in range(35):
            px = random.randint(0, w)
            py = random.randint(0, h)
            size = random.randint(1, 3)
            color = self.c_gold if random.random() > 0.5 else self.c_gold_dark
            p_id = self.canvas.create_oval(px, py, px+size, py+size, fill=color, outline="")
            vx = random.uniform(-0.2, 0.2)
            vy = random.uniform(-0.2, 0.2)
            self.particles.append([p_id, vx, vy])

    def draw_logo(self, cx, cy):
        size = 50
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.radians(angle_deg)
            px = cx + size * math.cos(angle_rad)
            py = cy + size * math.sin(angle_rad)
            points.extend([px, py])
            
        self.logo_base = self.canvas.create_polygon(points, outline=self.c_gold_dark, width=3, fill="")
        self.logo_glow = self.canvas.create_polygon(points, outline=self.c_gold, width=2, fill="")
        self.canvas.create_text(cx, cy, text="E", font=("Segoe UI", 32, "bold"), fill=self.c_gold)

    def draw_text(self, cx, cy):
        self.title_id = self.canvas.create_text(cx, cy + 20, text="TEAM ELMOLOK", 
                                                font=("Segoe UI", 42, "bold"), fill=self.c_gold)
        self.sub_id = self.canvas.create_text(cx, cy + 65, text="Advanced Image Processing Studio • Gold Edition", 
                                              font=("Segoe UI", 11), fill=self.c_text_sub)

    def draw_progress_bar(self, cx, cy, width):
        bh = 4 
        x1, y1 = cx - width/2, cy - bh/2
        x2, y2 = cx + width/2, cy + bh/2
        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1e293b", outline="")
        self.prog_bar_fill = self.canvas.create_rectangle(x1, y1, x1, y2, fill=self.c_gold, outline="")
        self.prog_tip = self.canvas.create_oval(x1-3, cy-3, x1+3, cy+3, fill="#ffffff", outline=self.c_gold, width=2)
        
        self.loading_txt_id = self.canvas.create_text(cx, cy + 25, text="Initializing Core Systems...", 
                                                      font=("Segoe UI", 9), fill=self.c_text_sub)
        self.bar_coords = (x1, y1, x2, y2, width)

    def fade_in_window(self):
        alpha = self.attributes("-alpha")
        if alpha < 1.0:
            alpha += 0.03
            self.attributes("-alpha", alpha)
            self.canvas.move(self.title_id, 0, -0.5)
            self.canvas.move(self.sub_id, 0, -0.5)
            self.after(20, self.fade_in_window)

    def animate_loop(self):
        # 1. Animate Particles
        for p in self.particles:
            p_id, vx, vy = p
            self.canvas.move(p_id, vx, vy)
            pos = self.canvas.coords(p_id)
            if pos: 
                if pos[0] < 0 or pos[2] > self.winfo_width(): p[1] = -vx
                if pos[1] < 0 or pos[3] > self.winfo_height(): p[2] = -vy

        # 2. Animate Logo Pulse
        self.pulse_size += 0.1 * self.pulse_direction
        if abs(self.pulse_size) > 1.5: self.pulse_direction *= -1
        new_width = 2 + self.pulse_size
        self.canvas.itemconfig(self.logo_glow, width=new_width)
        
        # 3. Animate Progress
        if self.loading_progress < 100:
            increment = 1.5 if self.loading_progress > 80 else 0.8
            self.loading_progress += increment
            
            # --- FIX: Default text provided first ---
            txt = "Initializing Core Systems..." 
            if self.loading_progress > 20: txt = "Loading GUI Framework..."
            if self.loading_progress > 50: txt = "Loading Image Processors..."
            if self.loading_progress > 85: txt = "Finalizing Setup..."
            if self.loading_progress >= 100: txt = "Ready. Launching..."
            
            self.canvas.itemconfig(self.loading_txt_id, text=txt)
            
            # Update bar
            x1, y1, x2, y2, total_w = self.bar_coords
            current_w = total_w * (self.loading_progress / 100)
            self.canvas.coords(self.prog_bar_fill, x1, y1, x1 + current_w, y2)
            
            # Update tip
            tip_x = x1 + current_w
            self.canvas.coords(self.prog_tip, tip_x-3, y1 + (y2-y1)/2 - 3, tip_x+3, y1 + (y2-y1)/2 + 3)

            self.after(30, self.animate_loop)
        else:
            self.after(500, self.start_app)

    def start_app(self):
        self.destroy()
        ImageProcessingApp().mainloop()


# ===================== Main App =====================
class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Studio – TEAM ELMOLOK")
        self.geometry("1280x800")
        try:
            self.state("zoomed") 
        except:
            pass # Fallback for non-Windows
        self.configure(bg=COLORS["bg"])

        self.image_gray = None
        self.image_rgb = None
        self.image2 = None

        self.setup_style()
        self.home_screen()

    def create_card(self, parent, icon_text, title, desc, cmd):
        # 1. Wrapper for Border
        wrapper = tk.Frame(parent, bg="#e2e8f0", padx=1, pady=1)
        wrapper.pack(side="left", padx=15, pady=10, expand=True, fill="both")
        
        # 2. Inner Content
        frame = tk.Frame(wrapper, bg="#ffffff", padx=25, pady=30)
        frame.pack(fill="both", expand=True)

        # Icons & Text
        tk.Label(frame, text=icon_text, font=("Segoe UI Emoji", 42), bg="#ffffff", fg="#4f46e5").pack(pady=(0, 15))
        tk.Label(frame, text=title, font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#1e293b").pack(pady=(0, 8))
        tk.Label(frame, text=desc, font=("Segoe UI", 11), bg="#ffffff", fg="#64748b", wraplength=220, justify="center").pack(pady=(0, 25))
        
        # Button
        btn_text = "View Credits" if "About" in title else "Launch Workspace"
        ttk.Button(frame, text=btn_text, style="Premium.TButton", command=cmd).pack(fill="x", padx=10)

        # 3. Hover Effect
        def on_enter(e): wrapper.config(bg="#4f46e5")
        def on_leave(e): wrapper.config(bg="#e2e8f0")
        frame.bind("<Enter>", on_enter)
        frame.bind("<Leave>", on_leave)

# ---------------- Style Engine ----------------
    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Layout Styles
        style.configure("Main.TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["card_bg"], relief="flat")
        
        # Typography
        style.configure("Title.TLabel", font=FONT_TITLE, background=COLORS["bg"], foreground=COLORS["text_main"])
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background=COLORS["card_bg"], foreground=COLORS["text_main"])
        style.configure("Sub.TLabel", font=FONT_SUB, background=COLORS["bg"], foreground=COLORS["text_sub"])
        style.configure("CardSub.TLabel", font=FONT_SUB, background=COLORS["card_bg"], foreground=COLORS["text_sub"])
        
        # --- BUTTONS ---
        style.configure(
            "Premium.TButton",
            font=FONT_BTN,
            background=COLORS["primary"],
            foreground="white",
            borderwidth=0,
            focuscolor=COLORS["bg"],
            padding=(20, 12)
        )
        style.map("Premium.TButton",
            background=[("active", COLORS["primary_hover"]), ("pressed", "#312e81")],
            relief=[("pressed", "sunken")]
        )

        style.configure(
            "Danger.TButton",
            font=FONT_BTN,
            background=COLORS["accent"],
            foreground="white",
            borderwidth=0,
            padding=(20, 12)
        )
        style.map("Danger.TButton", background=[("active", "#b91c1c")])

        # Ghost Button
        style.configure(
            "Ghost.TButton",
            font=FONT_BTN,
            background="#e2e8f0",
            foreground=COLORS["text_main"],
            borderwidth=0,
            padding=(15, 10)
        )
        style.map("Ghost.TButton", background=[("active", "#cbd5e1")])

        # Tabs
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", 
                        font=("Segoe UI", 11, "bold"), 
                        padding=[20, 12], 
                        background="#cbd5e1", 
                        foreground="#475569",
                        borderwidth=0)
        style.map("TNotebook.Tab", 
                  background=[("selected", COLORS["primary"]), ("active", "#94a3b8")],
                  foreground=[("selected", "white")])

    # ---------------- Scrollable Tab Logic ----------------
    def scrollable_tab(self, notebook, title):
        container = ttk.Frame(notebook, style="Main.TFrame")
        notebook.add(container, text=title)

        canvas = tk.Canvas(container, highlightthickness=0, bg=COLORS["bg"])
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Main.TFrame")

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def resize_frame(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", resize_frame)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        grid_frame = ttk.Frame(scroll_frame, style="Main.TFrame")
        grid_frame.pack(fill="x", padx=50, pady=20)
        return grid_frame

    # ---------------- Home ----------------
    def home_screen(self):
            self.clear()
            
            center_frame = tk.Frame(self, bg=COLORS["bg"])
            center_frame.place(relx=0.5, rely=0.5, anchor="center")

            # Header
            ttk.Label(center_frame, text="👑 Image Processing Studio 👑", style="Title.TLabel").pack(pady=(0, 10))
            ttk.Label(center_frame, text="Select a workspace to begin", style="Sub.TLabel").pack(pady=(0, 30))

            # Cards Container
            cards_box = tk.Frame(center_frame, bg=COLORS["bg"])
            cards_box.pack()

            # --- CARD 1: SINGLE ---
            self.create_card(cards_box, "🖼️", "Single Image", 
                            "Advanced filters, morphology, noise generation, and histogram analysis.", 
                            self.single_image_screen)
            
            # --- CARD 2: MULTI ---
            self.create_card(cards_box, "📑", "Multi-Layer", 
                            "Arithmetic operations between two images (Add, Subtract).", 
                            self.two_image_screen)

            # --- CARD 3: ABOUT (NEW) ---
            self.create_card(cards_box, "👥", "About Team", 
                            "Project credits, supervisors, and development team information.", 
                            self.show_about_dialog)

            tk.Label(center_frame, text="Powered by Team Elmolok", 
                    bg=COLORS["bg"], fg="#94a3b8", font=("Segoe UI", 9)).pack(pady=(40, 0))

    # ---------------- About Dialog (New & Professional) ----------------
    def show_about_dialog(self):
        about = tk.Toplevel(self)
        about.title("About Project")
        about.geometry("600x550")
        about.configure(bg=COLORS["splash_bg"])
        
        # Center the popup
        x = self.winfo_x() + (self.winfo_width() // 2) - 300
        y = self.winfo_y() + (self.winfo_height() // 2) - 275
        about.geometry(f"600x550+{x}+{y}")
        about.grab_set() # Modal behavior

        # Content
        tk.Label(about, text="👑 TEAM ELMOLOK 👑", font=("Segoe UI", 28, "bold"), 
                 bg=COLORS["splash_bg"], fg=COLORS["gold"]).pack(pady=(40, 10))
        
        tk.Label(about, text="Submitted Successfully To", font=("Segoe UI", 12, "bold"), 
                 bg=COLORS["splash_bg"], fg="#94a3b8").pack(pady=(20, 10))

        # Supervisors
        tk.Label(about, text="Dr. Marian Wagdy", font=("Segoe UI", 18, "bold"), 
                 bg=COLORS["splash_bg"], fg="white").pack()
        tk.Label(about, text="Eng. Nayera", font=("Segoe UI", 16, "bold"), 
                 bg=COLORS["splash_bg"], fg="white").pack()

        # Divider
        tk.Frame(about, height=2, bg=COLORS["gold"], width=400).pack(pady=10)

        # Team
        tk.Label(about, text="Executed By", font=("Segoe UI", 14, "bold"), 
                 bg=COLORS["splash_bg"], fg="#94a3b8").pack(pady=(15, 10))

        team_names = [
            "Mohamed Mahfouz",
            "Abdallah Khaled",
            "Bassem Mohamed",
            "Mohamed Omar",
            "Mohamed Abdelmonem"
        ]

        for name in team_names:
            tk.Label(about, text=f"• {name}", font=("Segoe UI", 13), 
                     bg=COLORS["splash_bg"], fg="white").pack(pady=2)

        # Close Button
        ttk.Button(about, text="Close", style="Danger.TButton", command=about.destroy).pack(pady=30)


    # ---------------- Single Image Screen ----------------
    def single_image_screen(self):
        self.clear()
        nav_bar = tk.Frame(self, bg="white", height=60, padx=20)
        nav_bar.pack(fill="x")
        
        ttk.Button(nav_bar, text="⬅ Home", style="Premium.TButton", command=self.home_screen).pack(side="left", pady=10)
        ttk.Label(nav_bar, text="Single Image Editor", font=("Segoe UI", 14, "bold"), background="white", foreground=COLORS["text_main"]).pack(side="left", padx=20, pady=15)
        ttk.Button(nav_bar, text="⬆ Upload New Image", style="Premium.TButton", command=self.upload_single).pack(side="right", pady=10)

        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=30, pady=30)

        self.brightness_tab(notebook)
        self.histogram_tab(notebook)
        self.filter_tab(notebook)
        self.noise_tab(notebook)
        self.morphology_tab(notebook)
        self.segmentation_tab(notebook)
        self.dither_tab(notebook)

    # ---------------- Tabs Generation ----------------
    def brightness_tab(self, nb):
        t = self.scrollable_tab(nb, "☀ Brightness & Solar")
        self.add_section_label(t, "Basic Operations")
        self.ask_btn(t, "Add Constant (+)", ops.brightness_add)
        self.ask_btn(t, "Subtract Constant (-)", ops.brightness_subtract)
        self.ask_btn(t, "Multiply Constant (×)", ops.brightness_multiply)
        self.ask_btn(t, "Divide Constant (÷)", ops.brightness_divide)
        
        self.add_section_label(t, "Effects")
        self.run_btn(t, "Invert Image (Negative)", lambda: ops.image_complement(self.image_gray))
        self.ask_btn(t, "Solarization Effect", ops.Solarization)

    def histogram_tab(self, nb):
        t = self.scrollable_tab(nb, "📊 Histogram Tools")
        self.add_section_label(t, "Grayscale Analysis")
        self.run_btn(t, "Show Histogram", lambda: ops.image_histogram(self.image_gray))
        self.run_btn(t, "Stretching", lambda: ops.histogram_stretching(self.image_gray))
        self.run_btn(t, "Equalization", lambda: ops.histogram_equalization(self.image_gray))

        self.add_section_label(t, "RGB Analysis")
        self.run_btn(t, "Show RGB Histogram", lambda: ops.rgb_image_histogram(self.image_rgb))
        self.run_btn(t, "RGB Stretching", lambda: ops.rgb_histogram_stretching(self.image_rgb))
        self.run_btn(t, "RGB Equalization", lambda: ops.rgb_histogram_equalization(self.image_rgb))

    def filter_tab(self, nb):
        t = self.scrollable_tab(nb, "🧹 Filters & Smoothing")
        self.add_section_label(t, "Linear Filters")
        self.run_btn(t, "Mean Filter", lambda: ops.mean_filter(self.image_gray))
        self.run_btn(t, "Gaussian Smoothing", lambda: ops.gaussian_smoothing(self.image_gray))
        
        self.add_section_label(t, "Non-Linear Filters")
        self.run_btn(t, "Median Filter", lambda: ops.median_filter(self.image_gray))
        self.run_btn(t, "Min Filter", lambda: ops.min_filter(self.image_gray))
        self.run_btn(t, "Max Filter", lambda: ops.max_filter(self.image_gray))
        self.run_btn(t, "Mode Filter", lambda: ops.mode_filter(self.image_gray))
        self.run_btn(t, "Range Filter", lambda: ops.range_filter(self.image_gray))
        
        self.add_section_label(t, "Edge Detection")
        self.run_btn(t, "Laplacian Filter", lambda: ops.laplacian_filtering(self.image_gray))

    def noise_tab(self, nb):
        t = self.scrollable_tab(nb, "📉 Noise Generator")
        self.run_btn(t, "Add Salt & Pepper Noise", lambda: ops.salt_pepper_noise(self.image_gray))
        self.ask_btn(t, "Add Gaussian Noise", ops.gaussian_noise)
        self.ask_btn(t, "Add Periodic Noise", ops.periodic_noise)

    def morphology_tab(self, nb):
        t = self.scrollable_tab(nb, "🧬 Morphology")
        self.add_section_label(t, "Basic Operations")
        self.ask_btn(t, "Dilation", ops.dilation_operation)
        self.ask_btn(t, "Erosion", ops.erosion_operation)
        
        self.add_section_label(t, "Compound Operations")
        self.ask_btn(t, "Opening", ops.opening_operation)
        self.ask_btn(t, "Closing", ops.closing_operation)
        self.ask_btn(t, "Apply All (Matrix)", ops.morphology_all_operations)

    def segmentation_tab(self, nb):
        t = self.scrollable_tab(nb, "📐 Segmentation")
        self.run_btn(t, "Otsu Thresholding", lambda: ops.otsu_segmentation(self.image_gray))

    def dither_tab(self, nb):
        t = self.scrollable_tab(nb, "🖨 Halftoning")
        self.run_btn(t, "Floyd–Steinberg Dithering", self.run_dither)

    # ---------------- UI Helpers ----------------
    def add_section_label(self, parent, text):
        ttk.Label(parent, text=text, font=("Segoe UI", 11, "bold"), foreground=COLORS["primary"], background=COLORS["bg"]).pack(fill="x", pady=(20, 5))
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=(0, 10))

    def run_btn(self, parent, text, cmd):
        ttk.Button(parent, text=text, style="Premium.TButton", command=lambda: self.safe_run(cmd)).pack(fill="x", pady=4)

    def ask_btn(self, parent, text, func):
        ttk.Button(parent, text=text, style="Premium.TButton", command=lambda: self.ask_and_run(func)).pack(fill="x", pady=4)

    # ---------------- Two Images Screen ----------------
    def two_image_screen(self):
        self.clear()
        header = tk.Frame(self, bg="white", height=70)
        header.pack(fill="x")
        ttk.Button(header, text="⬅ Back", style="Premium.TButton", command=self.home_screen).pack(side="left", padx=20, pady=15)
        ttk.Label(header, text="Dual Image Operations", style="Header.TLabel", background="white").pack(side="left", pady=15)

        workspace = tk.Frame(self, bg=COLORS["bg"])
        workspace.pack(expand=True, fill="both", padx=50, pady=30)
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=1)

        # LEFT CARD
        card1 = tk.Frame(workspace, bg=COLORS["card_bg"], padx=20, pady=20)
        card1.grid(row=0, column=0, padx=15, sticky="nsew")
        ttk.Label(card1, text="Image Source 1", style="Header.TLabel").pack(pady=10)
        ttk.Label(card1, text="Supports: JPG, PNG, JPEG", style="Sub.TLabel", background=COLORS["card_bg"]).pack(pady=(0, 20))
        self.status_img1 = ttk.Label(card1, text="No Image Selected", background=COLORS["card_bg"], foreground="#94a3b8")
        self.status_img1.pack(pady=5)
        ttk.Button(card1, text="📁 Select Image 1", style="Premium.TButton", command=lambda: self.upload_two(1)).pack(pady=20, fill="x")

        # RIGHT CARD
        card2 = tk.Frame(workspace, bg=COLORS["card_bg"], padx=20, pady=20)
        card2.grid(row=0, column=1, padx=15, sticky="nsew")
        ttk.Label(card2, text="Image Source 2", style="Header.TLabel").pack(pady=10)
        ttk.Label(card2, text="Must be same size for best results", style="Sub.TLabel", background=COLORS["card_bg"]).pack(pady=(0, 20))
        self.status_img2 = ttk.Label(card2, text="No Image Selected", background=COLORS["card_bg"], foreground="#94a3b8")
        self.status_img2.pack(pady=5)
        ttk.Button(card2, text="📁 Select Image 2", style="Premium.TButton", command=lambda: self.upload_two(2)).pack(pady=20, fill="x")

        # ACTION BAR
        action_bar = tk.Frame(self, bg=COLORS["bg"], height=100)
        action_bar.pack(fill="x", side="bottom", pady=40)
        ttk.Label(action_bar, text="Combine Operations", font=("Segoe UI", 12, "bold"), background=COLORS["bg"], foreground=COLORS["text_sub"]).pack(pady=10)
        btn_frame = tk.Frame(action_bar, bg=COLORS["bg"])
        btn_frame.pack()
        ttk.Button(btn_frame, text="➕ Add Images", style="Premium.TButton", width=25, command=self.run_add).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="➖ Subtract Images", style="Danger.TButton", width=25, command=self.run_subtract).pack(side="left", padx=10)

    # ---------------- Logic Wrappers ----------------
    def upload_single(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            img = Image.open(path)
            self.image_rgb = np.array(img.convert("RGB"))
            self.image_gray = np.array(img.convert("L"))
            messagebox.showinfo("Success", "Image loaded successfully.\nReady for processing.")

    def upload_two(self, idx):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path: return
        filename = path.split("/")[-1]
        img = np.array(Image.open(path).convert("RGB"))
        if idx == 1:
            self.image_rgb = img
            self.status_img1.config(text=f"✅ {filename}", foreground="green")
        else:
            self.image2 = img
            self.status_img2.config(text=f"✅ {filename}", foreground="green")

    def ask_and_run(self, func):
        if self.image_gray is None:
            messagebox.showerror("Attention", "Please upload an image first.")
            return
        v = simpledialog.askinteger("Parameter Required", "Enter value for operation:")
        if v is None: return
        func(self.image_gray, v)

    def safe_run(self, func):
        if self.image_gray is None:
            messagebox.showerror("Attention", "Please upload an image first.")
            return
        func()

    def run_add(self):
        if self.image_rgb is None or self.image2 is None:
            messagebox.showerror("Missing Data", "Please upload both Image 1 and Image 2.")
            return
        ops.add_two_images(self.image_rgb, self.image2)

    def run_subtract(self):
        if self.image_rgb is None or self.image2 is None:
            messagebox.showerror("Missing Data", "Please upload both Image 1 and Image 2.")
            return
        ops.subtract_two_images(self.image_rgb, self.image2)

    def run_dither(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            ops.floyd_steinberg_dithering(path)

    def clear(self):
        for w in self.winfo_children():
            w.destroy()

# ===================== Run =====================
if __name__ == "__main__":
    SplashScreen().mainloop()