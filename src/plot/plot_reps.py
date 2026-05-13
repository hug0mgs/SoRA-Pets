import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

def extract_label_from_filename(yaml_filename: str) -> str:
    """Extrai um label limpo do nome do arquivo (ex: paca12_pld0)"""
    name = Path(yaml_filename).stem
    if name.startswith("training_metrics_"):
        name = name[len("training_metrics_"):]
    return name.replace("_", " ").replace("paca", "PaCA").replace("pld", " PLD")

def plot_comparative_metrics(yaml_filenames=None, 
                            plot_std: bool = True, 
                            output_prefix: str = "comparison"):
    """
    Gera gráficos comparativos abrangentes (Acurácia, Perda, RAM, Energia, Camadas).
    """
    if yaml_filenames is None:
        yaml_filenames = sorted([
            f for f in os.listdir("src/plot") 
            if f.startswith("training_metrics_") and f.endswith(".yml")
        ])
        print(f"🔍 Encontrados {len(yaml_filenames)} arquivos de métricas.")

    if not yaml_filenames:
        print("❌ Nenhum arquivo YAML encontrado em 'src/plot/'")
        return

    dados_por_config = {}

    for yaml_file in yaml_filenames:
        yaml_path = f"src/plot/{yaml_file}" if not yaml_file.startswith("src/plot/") else yaml_file
        label = extract_label_from_filename(yaml_file)

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                history = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Erro ao ler {yaml_file}: {e}")
            continue

        metrics_agg = defaultdict(list)

        for _, metric_list in history.items():
            metrics_agg["times"].append([h.get("train_time_cum", 0) for h in metric_list])
            metrics_agg["accs"].append([h.get("accuracy", 0) * 100 for h in metric_list])
            metrics_agg["losses"].append([h.get("loss", 0) for h in metric_list])
            metrics_agg["ram"].append([h.get("ram_mb", 0) for h in metric_list])
            metrics_agg["energy"].append([h.get("energy_kwh", 0) for h in metric_list])
            metrics_agg["layers"].append([h.get("num_layers_active", 12) for h in metric_list])
            metrics_agg["gpu_util"].append([h.get("gpu_util_pct", 0) for h in metric_list])

        processed = {}
        for key, vals in metrics_agg.items():
            arr = np.array(vals)
            processed[f"{key}_mean"] = arr.mean(axis=0)
            processed[f"{key}_std"] = arr.std(axis=0)

        dados_por_config[label] = processed

    # Estilo visual
    plt.rcParams.update({
        'font.size': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'figure.figsize': (10, 6)
    })

    metrics_to_plot = [
        ("accs", "Test Accuracy (%)", "Accuracy vs Training Time"),
        ("losses", "Train Loss", "Loss vs Training Time"),
        ("ram", "RAM Usage (MB)", "RAM Usage vs Training Time"),
        ("energy", "Energy Consumed (kWh)", "Energy Consumption vs Training Time"),
        ("layers", "Active Layers", "Dynamic Layer Usage (PLD)"),
        ("gpu_util", "GPU Utilization (%)", "GPU Load vs Training Time")
    ]

    for key, ylabel, title in metrics_to_plot:
        plt.figure()
        for label, dados in dados_por_config.items():
            times = dados["times_mean"]
            mean = dados[f"{key}_mean"]
            std = dados[f"{key}_std"]
            
            plt.plot(times, mean, marker='o', markersize=4, label=label)
            if plot_std:
                plt.fill_between(times, mean - std, mean + std, alpha=0.15)

        plt.xlabel("Cumulative Training Time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best', fontsize='small')
        
        fname = f"src/plot/{output_prefix}_{key}.pdf"
        plt.savefig(fname, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

    # Gráfico de Eficiência: Acurácia Final / Energia Total
    plt.figure()
    labels = []
    efficiencies = []
    for label, dados in dados_por_config.items():
        energy_total = dados["energy_mean"][-1]
        acc_final = dados["accs_mean"][-1]
        if energy_total > 0:
            labels.append(label)
            efficiencies.append(acc_final / energy_total)
    
    if efficiencies:
        plt.bar(labels, efficiencies, color='skyblue', edgecolor='navy')
        plt.ylabel("Accuracy / kWh")
        plt.title("Hardware Efficiency Comparison")
        plt.xticks(rotation=30, ha='right')
        plt.savefig(f"src/plot/{output_prefix}_efficiency.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ Gráficos gerados com sucesso em 'src/plot/'.")

if __name__ == "__main__":
    plot_comparative_metrics()
