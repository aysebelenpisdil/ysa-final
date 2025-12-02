"""
Yapay Sinir Aglari - Sigara Icme Durumu Siniflandirmasi GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')


class SmokingClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YSA - Sigara Icme Durumu Siniflandirmasi")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Veri degiskenleri
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.model = None
        self.data_loaded = False

        # Model parametreleri
        self.hidden1 = tk.IntVar(value=128)
        self.hidden2 = tk.IntVar(value=64)
        self.max_iter = tk.IntVar(value=500)

        self.create_widgets()

    def create_widgets(self):
        # Ana cerceve
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Baslik
        title_label = ttk.Label(main_frame, text="YSA - Sigara Icme Durumu Siniflandirmasi",
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=10)

        # Ust panel - Veri ve Parametreler
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # Sol panel - Veri Yukleme
        data_frame = ttk.LabelFrame(top_frame, text="Veri Seti", padding="10")
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.load_btn = ttk.Button(data_frame, text="Veri Setini Yukle", command=self.load_data)
        self.load_btn.pack(pady=5)

        self.data_label = ttk.Label(data_frame, text="Veri yuklenmedi", foreground='red')
        self.data_label.pack(pady=5)

        # Sag panel - Model Parametreleri
        param_frame = ttk.LabelFrame(top_frame, text="Model Parametreleri", padding="10")
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Gizli katman 1
        ttk.Label(param_frame, text="Gizli Katman 1:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.hidden1, width=10).grid(row=0, column=1, pady=2)

        # Gizli katman 2
        ttk.Label(param_frame, text="Gizli Katman 2:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.hidden2, width=10).grid(row=1, column=1, pady=2)

        # Max iterasyon
        ttk.Label(param_frame, text="Max Iterasyon:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.max_iter, width=10).grid(row=2, column=1, pady=2)

        # Topoloji gosterimi
        self.topo_label = ttk.Label(param_frame, text="Topoloji: 25 -> 128 -> 64 -> 2", font=('Helvetica', 10, 'bold'))
        self.topo_label.grid(row=3, column=0, columnspan=2, pady=10)

        # Orta panel - Senaryo Secimi
        scenario_frame = ttk.LabelFrame(main_frame, text="Senaryo Secimi", padding="10")
        scenario_frame.pack(fill=tk.X, pady=10)

        self.scenario_var = tk.IntVar(value=1)

        scenarios = [
            ("Senaryo 1: Egitim = Test", 1),
            ("Senaryo 2: 5-Fold Cross Validation", 2),
            ("Senaryo 3: 10-Fold Cross Validation", 3),
            ("Senaryo 4: %75-25 Ayirma (5 farkli seed)", 4)
        ]

        for text, value in scenarios:
            ttk.Radiobutton(scenario_frame, text=text, variable=self.scenario_var,
                          value=value).pack(anchor=tk.W, pady=2)

        # Calistir butonu
        self.run_btn = ttk.Button(scenario_frame, text="Modeli Egit ve Test Et",
                                 command=self.run_scenario, style='Accent.TButton')
        self.run_btn.pack(pady=10)

        # Alt panel - Sonuclar
        result_frame = ttk.LabelFrame(main_frame, text="Sonuclar", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Sol - Metin sonuclari
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(text_frame, height=15, width=45, font=('Courier', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sag - Grafik
        self.graph_frame = ttk.Frame(result_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ag gorsellestirme butonu
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Ag Yapisini Goster", command=self.show_network).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Tum Senaryolari Karsilastir", command=self.compare_all).pack(side=tk.LEFT, padx=5)

    def load_data(self):
        try:
            # Veri setini yukle
            df = pd.read_csv('data/smoking.csv')

            # ID sutununu kaldir
            df = df.drop('ID', axis=1)

            # Kategorik degiskenleri encode et
            le = LabelEncoder()
            categorical_cols = ['gender', 'oral', 'tartar']
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])

            # X ve y ayir
            self.X = df.drop('smoking', axis=1).values
            self.y = df['smoking'].values
            self.X_scaled = self.scaler.fit_transform(self.X)

            self.data_loaded = True
            self.data_label.config(text=f"Yuklendi: {len(self.X):,} ornek, {self.X.shape[1]} ozellik",
                                  foreground='green')

            # Sinif dagilimini goster
            class_counts = np.bincount(self.y)
            info = f"Sinif Dagilimi:\n"
            info += f"  Sigara Icmeyen (0): {class_counts[0]:,}\n"
            info += f"  Sigara Icen (1): {class_counts[1]:,}\n"
            info += f"\nToplam: {len(self.y):,} ornek"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, info)

            self.update_topology_label()

            messagebox.showinfo("Basarili", "Veri seti basariyla yuklendi!")

        except Exception as e:
            messagebox.showerror("Hata", f"Veri yuklenirken hata: {str(e)}")

    def create_model(self, random_state=42):
        return MLPClassifier(
            hidden_layer_sizes=(self.hidden1.get(), self.hidden2.get()),
            activation='relu',
            solver='adam',
            max_iter=self.max_iter.get(),
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def update_topology_label(self):
        if self.X is not None:
            topo = f"Topoloji: {self.X.shape[1]} -> {self.hidden1.get()} -> {self.hidden2.get()} -> 2"
            self.topo_label.config(text=topo)

    def run_scenario(self):
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        self.update_topology_label()
        scenario = self.scenario_var.get()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Model egitiliyor...\n(Bu islem biraz zaman alabilir)\n")
        self.root.update()

        try:
            if scenario == 1:
                self.run_scenario1()
            elif scenario == 2:
                self.run_scenario2()
            elif scenario == 3:
                self.run_scenario3()
            elif scenario == 4:
                self.run_scenario4()
        except Exception as e:
            messagebox.showerror("Hata", f"Model egitilirken hata: {str(e)}")

    def run_scenario1(self):
        """Egitim = Test"""
        model = self.create_model()
        model.fit(self.X_scaled, self.y)
        y_pred = model.predict(self.X_scaled)
        acc = accuracy_score(self.y, y_pred)

        result = "SENARYO 1: Egitim = Test\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->2\n"
        result += f"Aktivasyon: ReLU\n"
        result += f"Optimizer: Adam\n\n"
        result += f"* Dogruluk: {acc*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 1")

    def run_scenario2(self):
        """5-Fold CV"""
        model = self.create_model()
        scores = cross_val_score(model, self.X_scaled, self.y, cv=5)
        y_pred = cross_val_predict(model, self.X_scaled, self.y, cv=5)

        result = "SENARYO 2: 5-Fold Cross Validation\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->2\n\n"
        result += "Fold Sonuclari:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i}: {score*100:.2f}%\n"
        result += f"\n* Ortalama: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 2 (5-Fold)")

    def run_scenario3(self):
        """10-Fold CV"""
        model = self.create_model()
        scores = cross_val_score(model, self.X_scaled, self.y, cv=10)
        y_pred = cross_val_predict(model, self.X_scaled, self.y, cv=10)

        result = "SENARYO 3: 10-Fold Cross Validation\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->2\n\n"
        result += "Fold Sonuclari:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i:2d}: {score*100:.2f}%\n"
        result += f"\n* Ortalama: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 3 (10-Fold)")

    def run_scenario4(self):
        """%75-25 Ayirma (5 seed)"""
        seeds = [42, 123, 456, 789, 999]
        results = []

        result = "SENARYO 4: %75-25 Ayirma (5 Seed)\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->2\n\n"

        best_acc = 0
        best_data = None

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.25, random_state=seed, stratify=self.y
            )
            model = self.create_model(random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append(acc)
            result += f"  Seed {seed}: {acc*100:.2f}%\n"

            if acc > best_acc:
                best_acc = acc
                best_data = (y_test, y_pred)

        avg = np.mean(results)
        std = np.std(results)
        result += f"\n* Ortalama: {avg*100:.2f}%\n"
        result += f"  Std: +/-{std*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(best_data[0], best_data[1], "Senaryo 4 (En Iyi)")

    def plot_confusion_matrix(self, y_true, y_pred, title):
        # Onceki grafigi temizle
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Icmiyor', 'Iciyor'])
        ax.set_yticklabels(['Icmiyor', 'Iciyor'])

        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                              color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=11)

        ax.set_xlabel('Tahmin')
        ax.set_ylabel('Gercek')
        ax.set_title(f'Konfuzyon Matrisi\n{title}')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def show_network(self):
        """Ag yapisini gorsellestir"""
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        # Yeni pencere
        net_window = tk.Toplevel(self.root)
        net_window.title("Yapay Sinir Agi Mimarisi")
        net_window.geometry("700x500")

        fig, ax = plt.subplots(figsize=(8, 5))

        layers = [self.X.shape[1], self.hidden1.get(), self.hidden2.get(), 2]
        layer_names = [f'Giris\n({layers[0]})', f'Gizli 1\n({layers[1]})',
                      f'Gizli 2\n({layers[2]})', f'Cikis\n({layers[3]})']
        colors = ['#3498db', '#2ecc71', '#2ecc71', '#e74c3c']

        x_positions = [0.15, 0.4, 0.65, 0.9]
        max_display = [8, 6, 6, 2]

        node_positions = []

        for i, (layer_size, max_d, x, color, name) in enumerate(zip(layers, max_display, x_positions, colors, layer_names)):
            positions = []
            n_display = min(layer_size, max_d)
            y_start = 0.5 + (n_display - 1) * 0.06

            for j in range(n_display):
                y = y_start - j * 0.12
                circle = plt.Circle((x, y), 0.025, color=color, ec='black', lw=2, zorder=10)
                ax.add_patch(circle)
                positions.append((x, y))

            if layer_size > max_d:
                ax.text(x, y_start - max_d * 0.12, '...', fontsize=16, ha='center', va='center')

            ax.text(x, 0.08, name, fontsize=10, ha='center', va='center', fontweight='bold')
            node_positions.append(positions)

        # Baglantilar
        for i in range(len(node_positions) - 1):
            for pos1 in node_positions[i]:
                for pos2 in node_positions[i + 1]:
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', alpha=0.3, lw=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Yapay Sinir Agi Mimarisi\nAktivasyon: ReLU | Optimizer: Adam', fontsize=12, fontweight='bold')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, net_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def compare_all(self):
        """Tum senaryolari karsilastir"""
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Tum senaryolar calistiriliyor...\n(Bu islem biraz zaman alabilir)\n")
        self.root.update()

        results = {}

        # Senaryo 1
        model1 = self.create_model()
        model1.fit(self.X_scaled, self.y)
        y_pred1 = model1.predict(self.X_scaled)
        results['Egitim=Test'] = accuracy_score(self.y, y_pred1)

        # Senaryo 2
        model2 = self.create_model()
        scores2 = cross_val_score(model2, self.X_scaled, self.y, cv=5)
        results['5-Fold CV'] = scores2.mean()

        # Senaryo 3
        model3 = self.create_model()
        scores3 = cross_val_score(model3, self.X_scaled, self.y, cv=10)
        results['10-Fold CV'] = scores3.mean()

        # Senaryo 4
        seeds = [42, 123, 456, 789, 999]
        accs = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.25, random_state=seed, stratify=self.y
            )
            model4 = self.create_model(random_state=seed)
            model4.fit(X_train, y_train)
            accs.append(accuracy_score(y_test, model4.predict(X_test)))
        results['%75-25'] = np.mean(accs)

        # Sonuclari goster
        result = "TUM SENARYOLARIN KARSILASTIRMASI\n"
        result += "="*35 + "\n\n"
        for name, acc in results.items():
            result += f"{name:<15}: {acc*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        # Grafik
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3.5))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
        bars = ax.bar(results.keys(), [v*100 for v in results.values()], color=colors)
        ax.set_ylabel('Dogruluk (%)')
        ax.set_title('Senaryo Karsilastirmasi')
        ax.set_ylim(70, 90)

        for bar, acc in zip(bars, results.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{acc*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

        plt.xticks(rotation=15, ha='right')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)


if __name__ == "__main__":
    root = tk.Tk()
    app = SmokingClassifierGUI(root)
    root.mainloop()
