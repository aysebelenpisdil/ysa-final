"""
Yapay Sinir Aglari - Sigara Icme Durumu Siniflandirmasi GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')


class YapaySinirAgi:
    """Yapay Sinir Agi Implementasyonu"""

    def __init__(self, katman_boyutlari, ogrenme_orani=0.1, iterasyon=100, random_state=42):
        self.katman_boyutlari = katman_boyutlari
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon = iterasyon
        self.random_state = random_state
        self.agirliklar = []
        self.biaslar = []
        self.loss_gecmisi = []

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_turev(self, x):
        return x * (1 - x)

    def mse_loss(self, y_gercek, y_tahmin):
        return np.mean((y_gercek - y_tahmin) ** 2)

    def agirlik_baslat(self):
        np.random.seed(self.random_state)
        self.agirliklar = []
        self.biaslar = []

        for i in range(len(self.katman_boyutlari) - 1):
            w = np.random.randn(self.katman_boyutlari[i], self.katman_boyutlari[i+1]) * 0.5
            b = np.zeros((1, self.katman_boyutlari[i+1]))
            self.agirliklar.append(w)
            self.biaslar.append(b)

    def ileri_yayilim(self, X):
        self.aktivasyonlar = [X]

        for i in range(len(self.agirliklar)):
            z = np.dot(self.aktivasyonlar[-1], self.agirliklar[i]) + self.biaslar[i]
            a = self.sigmoid(z)
            self.aktivasyonlar.append(a)

        return self.aktivasyonlar[-1]

    def geri_yayilim(self, X, y):
        m = X.shape[0]

        hata = self.aktivasyonlar[-1] - y
        deltalar = [hata * self.sigmoid_turev(self.aktivasyonlar[-1])]

        for i in range(len(self.agirliklar) - 1, 0, -1):
            hata = deltalar[0].dot(self.agirliklar[i].T)
            delta = hata * self.sigmoid_turev(self.aktivasyonlar[i])
            deltalar.insert(0, delta)

        for i in range(len(self.agirliklar)):
            self.agirliklar[i] -= self.ogrenme_orani * self.aktivasyonlar[i].T.dot(deltalar[i]) / m
            self.biaslar[i] -= self.ogrenme_orani * np.sum(deltalar[i], axis=0, keepdims=True) / m

    def egit(self, X, y, verbose=False):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.agirlik_baslat()
        self.loss_gecmisi = []

        for epoch in range(self.iterasyon):
            y_tahmin = self.ileri_yayilim(X)
            loss = self.mse_loss(y, y_tahmin)
            self.loss_gecmisi.append(loss)
            self.geri_yayilim(X, y)

        return self

    def tahmin(self, X):
        olasilik = self.ileri_yayilim(X)
        return (olasilik >= 0.5).astype(int).flatten()


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
        self.data_loaded = False

        # Model parametreleri
        self.hidden1 = tk.IntVar(value=64)
        self.hidden2 = tk.IntVar(value=32)
        self.ogrenme_orani = tk.DoubleVar(value=0.5)
        self.iterasyon = tk.IntVar(value=100)

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

        # Ogrenme orani
        ttk.Label(param_frame, text="Ogrenme Orani:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.ogrenme_orani, width=10).grid(row=2, column=1, pady=2)

        # Iterasyon
        ttk.Label(param_frame, text="Iterasyon:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.iterasyon, width=10).grid(row=3, column=1, pady=2)

        # Topoloji gosterimi
        self.topo_label = ttk.Label(param_frame, text="Topoloji: 25 -> 64 -> 32 -> 1", font=('Helvetica', 10, 'bold'))
        self.topo_label.grid(row=4, column=0, columnspan=2, pady=10)

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
            df = pd.read_csv('data/smoking.csv')
            df = df.drop('ID', axis=1)

            le = LabelEncoder()
            categorical_cols = ['gender', 'oral', 'tartar']
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])

            self.X = df.drop('smoking', axis=1).values
            self.y = df['smoking'].values
            self.X_scaled = self.scaler.fit_transform(self.X)

            self.data_loaded = True
            self.data_label.config(text=f"Yuklendi: {len(self.X):,} ornek, {self.X.shape[1]} ozellik",
                                  foreground='green')

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
        katman_boyutlari = [self.X.shape[1], self.hidden1.get(), self.hidden2.get(), 1]
        return YapaySinirAgi(
            katman_boyutlari=katman_boyutlari,
            ogrenme_orani=self.ogrenme_orani.get(),
            iterasyon=self.iterasyon.get(),
            random_state=random_state
        )

    def k_fold_cross_validation(self, X, y, k=5, random_state=42):
        np.random.seed(random_state)
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_size = n_samples // k
        scores = []
        tum_tahminler = np.zeros(n_samples)

        for fold in range(k):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k - 1 else n_samples
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = self.create_model(random_state=random_state + fold)
            model.egit(X_train, y_train, verbose=False)
            y_pred = model.tahmin(X_test)

            acc = np.mean(y_pred == y_test)
            scores.append(acc)
            tum_tahminler[test_indices] = y_pred

        return np.array(scores), tum_tahminler.astype(int)

    def update_topology_label(self):
        if self.X is not None:
            topo = f"Topoloji: {self.X.shape[1]} -> {self.hidden1.get()} -> {self.hidden2.get()} -> 1"
            self.topo_label.config(text=topo)

    def run_scenario(self):
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        self.update_topology_label()
        scenario = self.scenario_var.get()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Model egitiliyor...\n")
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
        model = self.create_model()
        model.egit(self.X_scaled, self.y)
        y_pred = model.tahmin(self.X_scaled)
        acc = np.mean(y_pred == self.y)

        result = "SENARYO 1: Egitim = Test\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->1\n"
        result += f"Aktivasyon: Sigmoid\n"
        result += f"Kayip: MSE\n\n"
        result += f"* Dogruluk: {acc*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 1")

    def run_scenario2(self):
        scores, y_pred = self.k_fold_cross_validation(self.X_scaled, self.y, k=5)

        result = "SENARYO 2: 5-Fold Cross Validation\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->1\n\n"
        result += "Fold Sonuclari:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i}: {score*100:.2f}%\n"
        result += f"\n* Ortalama: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 2 (5-Fold)")

    def run_scenario3(self):
        scores, y_pred = self.k_fold_cross_validation(self.X_scaled, self.y, k=10)

        result = "SENARYO 3: 10-Fold Cross Validation\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->1\n\n"
        result += "Fold Sonuclari:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i:2d}: {score*100:.2f}%\n"
        result += f"\n* Ortalama: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Senaryo 3 (10-Fold)")

    def run_scenario4(self):
        seeds = [42, 123, 456, 789, 999]
        results = []

        result = "SENARYO 4: %75-25 Ayirma (5 Seed)\n"
        result += "="*35 + "\n\n"
        result += f"Ag Topolojisi: {self.X.shape[1]}->{self.hidden1.get()}->{self.hidden2.get()}->1\n\n"

        best_acc = 0
        best_data = None

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.25, random_state=seed, stratify=self.y
            )
            model = self.create_model(random_state=seed)
            model.egit(X_train, y_train)
            y_pred = model.tahmin(X_test)
            acc = np.mean(y_pred == y_test)
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
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        net_window = tk.Toplevel(self.root)
        net_window.title("Yapay Sinir Agi Mimarisi")
        net_window.geometry("700x500")

        fig, ax = plt.subplots(figsize=(8, 5))

        layers = [self.X.shape[1], self.hidden1.get(), self.hidden2.get(), 1]
        layer_names = [f'Giris\n({layers[0]})', f'Gizli 1\n({layers[1]})',
                      f'Gizli 2\n({layers[2]})', f'Cikis\n({layers[3]})']
        colors = ['#3498db', '#2ecc71', '#2ecc71', '#e74c3c']

        x_positions = [0.15, 0.4, 0.65, 0.9]
        max_display = [8, 6, 6, 1]

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

        for i in range(len(node_positions) - 1):
            for pos1 in node_positions[i]:
                for pos2 in node_positions[i + 1]:
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', alpha=0.3, lw=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Yapay Sinir Agi Mimarisi\nAktivasyon: Sigmoid | Kayip: MSE', fontsize=12, fontweight='bold')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, net_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def compare_all(self):
        if not self.data_loaded:
            messagebox.showwarning("Uyari", "Once veri setini yukleyin!")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Tum senaryolar calistiriliyor...\n")
        self.root.update()

        results = {}

        # Senaryo 1
        model1 = self.create_model()
        model1.egit(self.X_scaled, self.y)
        y_pred1 = model1.tahmin(self.X_scaled)
        results['Egitim=Test'] = np.mean(y_pred1 == self.y)

        # Senaryo 2
        scores2, _ = self.k_fold_cross_validation(self.X_scaled, self.y, k=5)
        results['5-Fold CV'] = scores2.mean()

        # Senaryo 3
        scores3, _ = self.k_fold_cross_validation(self.X_scaled, self.y, k=10)
        results['10-Fold CV'] = scores3.mean()

        # Senaryo 4
        seeds = [42, 123, 456, 789, 999]
        accs = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.25, random_state=seed, stratify=self.y
            )
            model4 = self.create_model(random_state=seed)
            model4.egit(X_train, y_train)
            accs.append(np.mean(model4.tahmin(X_test) == y_test))
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
        ax.set_ylim(50, 85)

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
