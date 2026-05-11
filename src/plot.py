import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

def extract_label_from_filename(yaml_filename: str) -> str:
    """Extrai um label limpo do nome do arquivo (ex: paca12_pld0)"""
    name = Path(yaml_filename).stem
    # Remove prefixo comum
    if name.startswith("training_metrics_"):
        name = name[len("training_metrics_"):]
    return name.replace("_", " ").replace("paca", "PaCA").replace("pld", " PLD")


def plot_comparative_metrics(yaml_filenames=None, 
                            plot_std: bool = True, 
                            output_prefix: str = "comparison"):
    """
    Gera gráficos comparativos (Time × Accuracy e Time × Loss) 
    com UMA LINHA por arquivo YAML.
    
    Cada arquivo training_metrics_{paca}_pld{limit}.yml vira uma linha.
    """
    if yaml_filenames is None:
        # Auto-descobre todos os arquivos na pasta plot/
        yaml_filenames = sorted([
            f for f in os.listdir("plot") 
            if f.startswith("training_metrics_") and f.endswith(".yml")
        ])
        print(f"🔍 Encontrados {len(yaml_filenames)} arquivos de métricas automaticamente.")

    if not yaml_filenames:
        print("❌ Nenhum arquivo YAML encontrado na pasta 'plot/'")
        return

    # Dicionário para armazenar as curvas médias de cada configuração
    dados_por_config = {}

    for yaml_file in yaml_filenames:
        yaml_path = f"plot/{yaml_file}" if not yaml_file.startswith("plot/") else yaml_file
        label = extract_label_from_filename(yaml_file)

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                history = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Erro ao ler {yaml_file}: {e}")
            continue

        # Calcula média das repetições (_repN)
        times_list = []
        accs_list = []
        losses_list = []
        gens_list = []      # ← novo
        memos_list = []     # ← novo

        for _, metric_list in history.items():
            times = [h.get("train_time_cum", 0) for h in metric_list]
            accs = []
            losses = []
            gens= []
            memos = []
            for h in metric_list:
                acc = h.get("accuracy")
                if acc is not None:
                    acc = acc * 100 if acc <= 1.0 else acc
                accs.append(acc)
                losses.append(h.get("loss"))
                gen_val = h.get("gen")
                memo_val = h.get("memo")
                gens.append(gen_val if gen_val is not None else 0.0)
                memos.append(memo_val if memo_val is not None else 1.0)

            times_list.append(times)
            accs_list.append(accs)
            losses_list.append(losses)
            gens_list.append(gens)
            memos_list.append(memos)

        # Média por época
        times_arr = np.array(times_list)
        accs_arr = np.array(accs_list)
        losses_arr = np.array(losses_list)
        gens_arr = np.array(gens_list)
        memos_arr = np.array(memos_list)

        dados_por_config[label] = {
            "times": times_arr.mean(axis=0),
            "accs": accs_arr.mean(axis=0),
            "losses": losses_arr.mean(axis=0),
            "gens": gens_arr.mean(axis=0),           # ← novo
            "memos": memos_arr.mean(axis=0),         # ← novo
            "accs_std": accs_arr.std(axis=0) if plot_std else None,
            "losses_std": losses_arr.std(axis=0) if plot_std else None,
            "gens_std": gens_arr.std(axis=0) if plot_std else None,   # ← novo
            "memos_std": memos_arr.std(axis=0) if plot_std else None, # ← novo
        }

    if not dados_por_config:
        print("❌ Nenhum dado válido encontrado.")
        return

    # Estilo científico (artigo)
    plt.rcParams.update({
        'font.size': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'figure.autolayout': True
    })

    # ==========================================================
    # Gráfico 1: Cumulative Time × Accuracy
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for label, dados in dados_por_config.items():
        plt.plot(dados["times"], dados["accs"], marker='o', label=label)
        if plot_std and dados["accs_std"] is not None:
            plt.fill_between(dados["times"],
                             dados["accs"] - dados["accs_std"],
                             dados["accs"] + dados["accs_std"],
                             alpha=0.15)

    plt.xlabel("Cumulative Training Time (s)")
    plt.ylabel("Average Test Accuracy (%)")
    plt.ylim(85, 95)                    # ajuste conforme seus resultados reais
    plt.legend(loc="lower right", fontsize=10)
    plt.title("Comparison: Accuracy vs Training Time")
    plt.savefig(f"plot/{output_prefix}_Time_x_Acc.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # ==========================================================
    # Gráfico 2: Cumulative Time × Loss
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for label, dados in dados_por_config.items():
        plt.plot(dados["times"], dados["losses"], marker='o', label=label)
        if plot_std and dados["losses_std"] is not None:
            plt.fill_between(dados["times"],
                             dados["losses"] - dados["losses_std"],
                             dados["losses"] + dados["losses_std"],
                             alpha=0.15)

    plt.xlabel("Cumulative Training Time (s)")
    plt.ylabel("Average Train Loss")
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Comparison: Loss vs Training Time")
    plt.savefig(f"plot/{output_prefix}_Time_x_Loss.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # ==========================================================
    # Gráfico 3: Cumulative Time × Memorization Ratio (novo)
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for label, dados in dados_por_config.items():
        plt.plot(dados["times"], dados["memos"], marker='s', label=label)
        if plot_std and dados["memos_std"] is not None:
            plt.fill_between(dados["times"],
                             dados["memos"] - dados["memos_std"],
                             dados["memos"] + dados["memos_std"],
                             alpha=0.15)
    plt.xlabel("Cumulative Training Time (s)")
    plt.ylabel("Memorization Ratio (eval_acc / train_acc)")
    #plt.ylim(0.85, 1.05)          # típico para boa compressão
    plt.legend(loc="lower right", fontsize=10)
    plt.title("Memorization Ratio vs Training Time")
    plt.savefig(f"plot/{output_prefix}_Time_x_Memo.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # ==========================================================
    # Gráfico 4: Cumulative Time × Generalization Gap (novo)
    # ==========================================================
    plt.figure(figsize=(10, 6))
    for label, dados in dados_por_config.items():
        plt.plot(dados["times"], dados["gens"], marker='^', label=label)
        if plot_std and dados["gens_std"] is not None:
            plt.fill_between(dados["times"],
                             dados["gens"] - dados["gens_std"],
                             dados["gens"] + dados["gens_std"],
                             alpha=0.15)
    plt.xlabel("Cumulative Training Time (s)")
    plt.ylabel("Generalization Gap (train_acc - eval_acc)")
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Generalization Gap vs Training Time")
    plt.savefig(f"plot/{output_prefix}_Time_x_Gen.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("✅ Gráficos comparativos gerados com sucesso!")
    print(f" • plot/{output_prefix}_Time_x_Acc.pdf")
    print(f" • plot/{output_prefix}_Time_x_Loss.pdf")
    print(f" • plot/{output_prefix}_Time_x_Memo.pdf   ← novo")
    print(f" • plot/{output_prefix}_Time_x_Gen.pdf    ← novo")
    print(f" Configurações comparadas: {list(dados_por_config.keys())}")


if __name__ == "__main__":
    # === USO RECOMENDADO ===
    # 1. Comparar todos os arquivos automaticamente:
    plot_comparative_metrics()

    # 2. Ou escolher manualmente quais arquivos comparar:
    # plot_comparative_metrics([
    #     "training_metrics_paca12_pld0.yml",
    #     "training_metrics_paca6_pld3.yml",
    #     "training_metrics_paca-none_pld6.yml"
    # ], output_prefix="paca_vs_pld")