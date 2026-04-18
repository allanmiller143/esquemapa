# -*- coding: utf-8 -*-

import os
import json
import random
from turtle import st
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import shorten, wrap
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
from minisom import MiniSom
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib import cm

# ======================================================
# GLOBAL CONFIG
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "step3_output"
OUTPUT_DIR = DATA_DIR / "step5_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["OPENAI", "DEEPSEEK", "MISTRAL"]
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
EMB_CACHE = OUTPUT_DIR / "theme_embeddings.npy"
BEST_PARAMS_FILE = OUTPUT_DIR / "som_best_params.json"

# ======================================================
# UTILS
# ======================================================
def load_env_vars():
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada no arquivo .env")
    return api_key

def normalize_theme(text):
    if text is None:
        return ""
    text = str(text)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text.lower().strip()

def safe_list(x):
    return x if isinstance(x, list) else []

def _short(s, width=48):
    return shorten(str(s), width=width, placeholder="…")

# ======================================================
# LOAD DATA
# ======================================================
def load_all_topics():
    themes = set()
    occurrences = Counter()
    tables_by_theme = defaultdict(set)

    for model in MODELS:
        path = INPUT_DIR / f"table_topics_{model}_1.json"
        if not path.exists():
            print(f"Aviso: Arquivo não encontrado: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for table, topics in data.items():
            for t in safe_list(topics):
                nt = normalize_theme(t)
                if nt:
                    themes.add(nt)
                    occurrences[nt] += 1
                    tables_by_theme[nt].add(table)

    if not themes:
        raise ValueError("Nenhum tema foi carregado. Verifique os arquivos de entrada.")

    return sorted(themes), occurrences, tables_by_theme

# ======================================================
# EMBEDDINGS
# ======================================================
def get_embeddings(texts, api_key):
    if EMB_CACHE.exists():
        print(f"Carregando embeddings do cache: {EMB_CACHE}")
        return np.load(EMB_CACHE)

    print(f"Gerando embeddings para {len(texts)} temas...")
    client = OpenAI(api_key=api_key)
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])

    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(EMB_CACHE, embeddings)
    return embeddings

# ======================================================
# SOM METRICS
# ======================================================
def quantization_error(som, data):
    return som.quantization_error(data)

def topographic_error(som, data):
    return som.topographic_error(data)

# ======================================================
# GRID SEARCH
# ======================================================
def grid_search_som(data):
    print("Iniciando Grid Search para otimização do SOM...")

    grid = {
        "x": [18],
        "y": [16],
        "sigma": [2.5],
        "learning_rate": [0.1],
        "iterations": [2000],
    }

    best = None
    results = []
    total_comb = int(np.prod([len(v) for v in grid.values()]))
    curr = 0

    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        curr += 1

        som = MiniSom(
            params["x"], params["y"], data.shape[1],
            sigma=params["sigma"],
            learning_rate=params["learning_rate"],
            random_seed=SEED,
            topology="hexagonal"
        )

        som.random_weights_init(data)
        som.train_batch(data, params["iterations"], verbose=False)

        qe = float(quantization_error(som, data))
        te = float(topographic_error(som, data))
        score = qe + te

        entry = {**params, "qe": qe, "te": te, "score": score}
        results.append(entry)

        if best is None or score < best["score"]:
            best = entry

        if curr % 10 == 0 or curr == total_comb:
            print(f"\rProgresso: {curr}/{total_comb} combinações testadas.", end="")

    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"Melhores parâmetros salvos em: {BEST_PARAMS_FILE}")
    return best

# ======================================================
# U-MATRIX (HEX)
# ======================================================
def compute_umatrix_hexagonal(som):
    """
    Calcula a U-Matrix considerando vizinhança de 6 vizinhos (grade hexagonal).
    """
    weights = som.get_weights()
    x_dim, y_dim, _ = weights.shape
    umatrix = np.zeros((x_dim, y_dim), dtype=np.float32)

    for x in range(x_dim):
        for y in range(y_dim):
            if y % 2 == 0:
                adj = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
            else:
                adj = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]

            neighbors = []
            for dx, dy in adj:
                nx, ny = x + dx, y + dy
                if 0 <= nx < x_dim and 0 <= ny < y_dim:
                    neighbors.append(weights[nx, ny])

            if neighbors:
                umatrix[x, y] = float(np.mean([np.linalg.norm(weights[x, y] - n) for n in neighbors]))

    return umatrix

def plot_umatrix_hex_static(umatrix, out_png, macrothemes=None, top_n=10, cmap="inferno"):
    """
    Plot estático com células hexagonais em formato de colmeia perfeita.
    Colorbar com altura EXATAMENTE igual à colmeia.

    Args:
        umatrix: matriz U-Matrix
        out_png: caminho para salvar a figura
        macrothemes: lista de dicionários com macrotemas (opcional)
        top_n: número de macrotemas para anotar (padrão: 10)
        cmap: colormap (padrão: "inferno")
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x_dim, y_dim = umatrix.shape
    patches = []
    colors = []

    # Espaçamento correto para colmeia hexagonal flat-top
    hex_width = 1.0
    hex_height = np.sqrt(3) / 2 * hex_width

    # Mapear posições dos hexágonos
    hex_positions = {}

    for x in range(x_dim):
        for y in range(y_dim):
            # Offset horizontal para linhas ímpares (colmeia)
            x_offset = 0.5 * hex_width if (y % 2 == 1) else 0.0

            # Posição do centro do hexágono
            x_plot = x * hex_width + x_offset
            y_plot = y * hex_height

            hex_positions[(x, y)] = (x_plot, y_plot)

            # Criar hexágono com orientação 0 (flat-top) para colmeia perfeita
            hexagon = RegularPolygon(
                (x_plot, y_plot),
                numVertices=6,
                radius=hex_width / np.sqrt(3),
                orientation=0,
                facecolor='none',
                edgecolor='white',
                linewidth=0.5
            )

            patches.append(hexagon)
            colors.append(float(umatrix[x, y]))

    fig, ax = plt.subplots(figsize=(14, 12))

    # Adicionar hexágonos com cores
    pc = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidth=0.5)
    pc.set_array(np.asarray(colors, dtype=np.float32))
    ax.add_collection(pc)

    # Adicionar anotações dos top N macrotemas
    if macrothemes is not None and len(macrothemes) > 0:
        top_macros = macrothemes[:min(top_n, len(macrothemes))]

        for i, macro in enumerate(top_macros, 1):
            neuron_str = macro["neuron"]  # formato "x_y"
            x_coord, y_coord = map(int, neuron_str.split("_"))

            if (x_coord, y_coord) in hex_positions:
                x_plot, y_plot = hex_positions[(x_coord, y_coord)]

                # Texto do macrotema (encurtado)
                label = _short(macro["macrotema"], 25)

                # Adicionar número do ranking
                text = f"{i}. {label}"

                # Adicionar texto com fundo branco semi-transparente
                ax.text(
                    x_plot, y_plot, text,
                    fontsize=7,
                    ha='center', va='center',
                    weight='bold',
                    color='white',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='black',
                        edgecolor='white',
                        alpha=0.7,
                        linewidth=0.8
                    ),
                    zorder=10
                )

    # Configurar eixos
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")

    # Criar colorbar com altura EXATA do gráfico usando make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(pc, cax=cax, label="Distância média (U-Matrix)")

    # Título
    title = f"U-Matrix (grade hexagonal - colmeia)"
    if macrothemes:
        title += f"\nTop {min(top_n, len(macrothemes))} macrotemas"

    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figura salva: {out_png}")


def export_macrothemes_table(macrothemes, out_csv, top_n=40, max_subthemes=10):
    rows = []
    for i, m in enumerate(macrothemes[:top_n], 1):
        subthemes_str = "; ".join(m["subtemas"][:max_subthemes])
        if len(m["subtemas"]) > max_subthemes:
            subthemes_str += f"; ... ({len(m['subtemas']) - max_subthemes} mais)"
        rows.append({
            "Ranking": i,
            "Macrotema": m["macrotema"],
            "Frequência Total": m["frequencia_total"],
            "Número de Subtemas": m["num_subtemas"],
            "Neurônio": m["neuron"],
            "Subtemas (amostra)": subthemes_str
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Tabela de macrotemas exportada: {out_csv}")


def export_outliers_table(outliers, theme_to_neuron, occurrences, out_csv, top_n=40):
    rows = []
    for i, o in enumerate(outliers[:top_n], 1):
        rows.append({
            "Ranking": i,
            "Tema": o["tema"],
            "Quantization Error": round(o["qe"], 6),
            "Frequência": occurrences.get(o["tema"], 0),
            "Neurônio": theme_to_neuron.get(o["tema"], "-")
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Tabela de outliers exportada: {out_csv}")


def plot_top_macrothemes_bar(macrothemes, out_png, top_n=20):
    top = macrothemes[:top_n]
    names = [m["macrotema"][:30] for m in top]
    freqs = [m["frequencia_total"] for m in top]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(names, freqs, color="#ff9f43")
    ax.set_xlabel("Frequência Total", fontsize=11, weight='bold')
    ax.set_title(f"Top {top_n} Macrotemas por Frequência", fontsize=13, weight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de macrotemas salvo: {out_png}")


def export_cluster_theme_tables(macrothemes, tables_by_theme, occurrences, out_csv):
    rows = []
    for m in macrothemes:
        for theme in m["subtemas"]:
            tables = sorted(list(tables_by_theme.get(theme, [])))
            rows.append({
                "Macrotema": m["macrotema"],
                "Tema": theme,
                "Frequência": occurrences.get(theme, 0),
                "Tabelas": "; ".join(tables) if tables else "-"
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Tabela de temas e tabelas por cluster exportada: {out_csv}")


def export_tables_by_cluster(macrothemes, tables_by_theme, out_csv):
    rows = []
    for m in macrothemes:
        all_tables = set()
        for theme in m["subtemas"]:
            all_tables.update(tables_by_theme.get(theme, []))
        rows.append({
            "Macrotema": m["macrotema"],
            "Número de Tabelas": len(all_tables),
            "Tabelas": "; ".join(sorted(all_tables)) if all_tables else "-"
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Tabela de tabelas por cluster exportada: {out_csv}")


def compute_hits_matrix(x_dim, y_dim, winners):
    hits = np.zeros((x_dim, y_dim), dtype=int)
    for winner in winners:
        hits[winner[0], winner[1]] += 1
    return hits


def plot_hits_heatmap(hits, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(hits.T, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xlabel("X", fontsize=11, weight='bold')
    ax.set_ylabel("Y", fontsize=11, weight='bold')
    ax.set_title("Hits Matrix (Frequência de Ativação)", fontsize=13, weight='bold')
    plt.colorbar(im, ax=ax, label="Hits")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap de hits salvo: {out_png}")


def plot_scatter_winners(x_dim, y_dim, winners, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    xs = [w[0] for w in winners]
    ys = [w[1] for w in winners]
    ax.scatter(xs, ys, alpha=0.5, s=30, c="#ff9f43")
    ax.set_xlim(-0.5, x_dim - 0.5)
    ax.set_ylim(-0.5, y_dim - 0.5)
    ax.set_xlabel("X", fontsize=11, weight='bold')
    ax.set_ylabel("Y", fontsize=11, weight='bold')
    ax.set_title("Distribuição de Neurônios Vencedores", fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot de winners salvo: {out_png}")


def load_metadata_json(metadata_path):
    """
    Carrega metadata.json e indexa por table_name.
    """
    if not metadata_path.exists():
        print(f"Aviso: Arquivo de metadata não encontrado: {metadata_path}")
        return {}
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metadata_dict = {}
    for table_meta in data:
        table_name = table_meta.get("table_name", "")
        if table_name:
            metadata_dict[table_name] = table_meta
    
    return metadata_dict


# ======================================================
# MAPA INTERATIVO SOM (D3.JS) - V7 (ALINHAMENTO ESQUERDA FIXO)
# COM MODAL PARA DETALHES DE TABELAS - CONTEÚDO ENRIQUECIDO
# ======================================================
def build_interactive_map_v3(som, themes, embeddings, macrothemes, occurrences, tables_by_theme, out_path, umatrix, metadata_dict=None, cmap_name="inferno"):
    x_dim, y_dim = umatrix.shape
    neuron_data = defaultdict(list)
    for i, theme in enumerate(themes):
        win = som.winner(embeddings[i])
        tabelas = sorted(list(tables_by_theme.get(theme, [])))
        neuron_data[win].append({
            "theme": theme, 
            "freq": int(occurrences.get(theme, 0)),
            "tables": tabelas
        })

    grid_data = []
    u_min, u_max = umatrix.min(), umatrix.max()
    colormap = cm.get_cmap(cmap_name)
    
    def get_hex_color(val):
        norm_val = (val - u_min) / (u_max - u_min) if u_max > u_min else 0
        rgba = colormap(norm_val)
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    for x in range(x_dim):
        for y in range(y_dim):
            u_val = float(umatrix[x, y])
            themes_in_neuron = sorted(neuron_data.get((x, y), []), key=lambda x: x['freq'], reverse=True)
            macro = next((m for m in macrothemes if m['neuron'] == f"{x}_{y}"), None)
            
            has_themes = len(themes_in_neuron) > 0
            hex_color = get_hex_color(u_val) if has_themes else "transparent"
            
            grid_data.append({
                "id": f"{x}_{y}",
                "x": x, "y": y, "u_val": u_val,
                "color": hex_color, "count": len(themes_in_neuron),
                "themes": themes_in_neuron,
                "macro": macro['macrotema'] if macro else None
            })

    data_json = json.dumps(grid_data, ensure_ascii=False)
    macros_ordered = sorted(macrothemes, key=lambda x: x["frequencia_total"], reverse=True)
    macros_json = json.dumps([m['macrotema'] for m in macros_ordered[:20]], ensure_ascii=False)
    
    # Converter metadata_dict para JSON (se fornecido)
    metadata_json = json.dumps(metadata_dict or {}, ensure_ascii=False, default=str)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>Mapa Hexbin — Temas do SOM</title>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <style>
        :root {{
            --bg: #121212; --surface: #1e1e1e; --text: #e0e0e0; --text-muted: #a0a0a0;
            --primary: #ff9f43; --border: #333; --sidebar-bg: #181818;
        }}
        [data-theme="light"] {{
            --bg: #f5f5f5; --surface: #ffffff; --text: #222222; --text-muted: #666666;
            --primary: #e67e22; --border: #ddd; --sidebar-bg: #fdfdfd;
        }}
        body {{ margin: 0; font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); overflow: hidden; }}
        #app {{ width: 100vw; height: 100vh; display: flex; flex-direction: column; }}
        
        header {{ display: flex; justify-content: space-between; align-items: flex-start; padding: 30px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }}
        .title-area h1 {{ margin: 0; font-size: 1.3rem; font-weight: 700; }}
        .title-area p {{ margin: 3px 0 0; color: var(--text-muted); font-size: 0.85rem; }}
        
        .controls-top {{ display: flex; align-items: center; gap: 20px; padding: 12px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }}
        .control-group {{ display: flex; align-items: center; gap: 10px; font-size: 0.8rem; }}
        input[type="text"] {{ background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 5px 12px; border-radius: 20px; outline: none; width: 180px; font-size: 0.8rem; }}
        
        .main-content {{ display: flex; flex: 1; overflow: hidden; position: relative; }}
        
        #viz-container {{ flex: 1; background: var(--bg); position: relative; overflow: hidden; }}
        svg {{ width: 100%; height: 100%; cursor: default; }}
        
        #sidebar {{
            width: 320px; background: var(--sidebar-bg); border-left: 1px solid var(--border);
            display: flex; flex-direction: column; transition: transform 0.3s ease;
            position: relative; z-index: 100; flex-shrink: 0;
        }}
        .sidebar-header {{ padding: 20px; border-bottom: 1px solid var(--border); }}
        .sidebar-header h2 {{ margin: 0; font-size: 1.1rem; color: var(--primary); }}
        .sidebar-empty {{ display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); text-align: center; padding: 20px; font-style: italic; }}
        .theme-list-item {{ margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
        .theme-list-item:last-child {{ border-bottom: none; }}
        .theme-freq {{ color: var(--primary); font-weight: bold; font-size: 0.75rem; }}

        .hexagon {{ stroke: var(--bg); stroke-width: 0.5px; transition: filter 0.2s; cursor: pointer; pointer-events: all; }}
        .hexagon:hover {{ filter: brightness(5); }}
        .hexagon.highlight {{ filter: brightness(5); }}
        .hexagon.dimmed {{ opacity: 0.1; }}
        .hexagon.empty {{ stroke: var(--border); stroke-dasharray: 2,2; pointer-events: none; }}
        .hexagon.selected {{ filter: brightness(50) !important; stroke: var(--primary) !important; stroke-width: 2px !important; }}

        .axis-label {{ font-size: 10px; fill: var(--text-muted); pointer-events: none; }}
        .grid-line {{ stroke: var(--border); stroke-dasharray: 2,2; stroke-width: 0.5; pointer-events: none; }}
        
        #tooltip {{
            position: fixed; pointer-events: none; background: rgba(0,0,0,0.95);
            border: 1px solid #555; padding: 12px; border-radius: 8px; font-size: 0.8rem;
            display: none; box-shadow: 0 8px 20px rgba(0,0,0,0.7); z-index: 2000; color: #fff;
            max-height: 400px; overflow-y: auto; max-width: 300px;
        }}
        
        .t-macro {{ color: var(--primary); font-weight: bold; display: block; margin-bottom: 5px; border-bottom: 1px solid #444; padding-bottom: 4px; }}
        
        #legend {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 12px 20px; background: var(--surface); border-top: 1px solid var(--border); justify-content: flex-start; overflow-y: auto; flex-shrink: 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.7rem; color: var(--text-muted); background: var(--bg); padding: 5px 10px; border-radius: 20px; border: 1px solid var(--border); cursor: pointer; transition: all 0.2s; white-space: nowrap; }}
        .legend-item:hover {{ border-color: var(--primary); color: var(--text); }}
        .legend-item.active {{ background: var(--primary); color: #000; border-color: var(--primary); }}
        .legend-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
        .legend-section-title {{ width: 100%; font-size: 0.7rem; font-weight: 600; color: var(--text-muted); padding: 5px 0; text-transform: uppercase; letter-spacing: 0.5px; }}
        
        .stats-panel {{ position: absolute; top: 10px; right: 10px; background: var(--surface); border: 1px solid var(--border); padding: 10px; border-radius: 8px; font-size: 0.75rem; z-index: 10; pointer-events: none; }}
        .stat-item {{ margin-bottom: 4px; }}
        .stat-value {{ font-weight: bold; color: var(--primary); }}
        .export-btn {{ background: var(--primary); color: #000; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 0.75rem; margin-left: auto; }}
        .export-btn:hover {{ opacity: 0.9; }}
        .sidebar-search {{ padding: 10px 20px; border-bottom: 1px solid var(--border); }}
        .sidebar-search input {{ width: 100%; box-sizing: border-box; }}
        .table-tag {{ display: inline-block; background: var(--border); color: var(--text-muted); font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; margin: 2px; cursor: pointer; transition: all 0.2s; }}
        .table-tag:hover {{ background: var(--primary); color: #000; }}
        
        /* MODAL STYLES */
        .modal-overlay {{
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.7); z-index: 3000; align-items: center; justify-content: center;
            padding: 20px;
        }}
        .modal-overlay.active {{ display: flex; }}
        
        .modal-content {{
            background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
            max-width: 1100px; width: 95%; max-height: 88vh; overflow: hidden;
            padding: 0; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9);
            display: flex; flex-direction: column;
        }}
        
        .modal-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0; border-bottom: 1px solid var(--border); padding: 18px 22px;
            flex-shrink: 0; background: var(--surface);
        }}
        
        .modal-header h2 {{
            margin: 0; font-size: 1.3rem; color: var(--text); word-break: break-word;
            padding-right: 10px;
        }}
        
        .modal-close {{
            background: none; border: none; color: var(--text); font-size: 1.5rem;
            cursor: pointer; padding: 0; width: 30px; height: 30px; display: flex;
            align-items: center; justify-content: center; transition: color 0.2s;
            flex-shrink: 0;
        }}
        
        .modal-close:hover {{ color: var(--primary); }}

        #modal-body {{
            padding: 20px 22px 22px 22px;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        
        .modal-section {{
            margin-bottom: 22px;
        }}
        
        .modal-section h3 {{
            font-size: 1rem; color: var(--text); margin: 0 0 12px 0;
            border-left: none; padding-left: 0; font-weight: 600;
        }}
        
        .info-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 15px;
        }}
        
        .info-item {{
            background: var(--bg); padding: 12px; border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .info-label {{
            font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: none;
            letter-spacing: 0; margin-bottom: 4px;
        }}
        
        .info-value {{
            font-size: 0.9rem; color: var(--text); word-break: break-word;
        }}

        .table-wrapper {{
            width: 100%;
            overflow-x: auto;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg);
        }}
        
        .columns-table {{
            width: max-content; min-width: 100%;
            border-collapse: collapse; background: var(--bg);
        }}
        
        .columns-table th {{
            background: var(--surface); color: var(--text-muted); padding: 10px; text-align: left;
            font-weight: 600; font-size: 0.8rem; border-bottom: 1px solid var(--border);
            position: sticky; top: 0; z-index: 2;
            white-space: nowrap;
        }}
        
        .columns-table td {{
            padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.85rem;
            vertical-align: top; white-space: nowrap;
        }}
        
        .columns-table tr:hover {{
            background: rgba(255, 159, 67, 0.1);
        }}

        .columns-table th:first-child,
        .columns-table td:first-child {{
            position: sticky;
            left: 0;
            z-index: 1;
            background: var(--bg);
        }}

        .columns-table thead th:first-child {{
            z-index: 3;
            background: var(--surface);
        }}
        
        .stat-badge {{
            display: inline-block; background: transparent; color: var(--text-muted);
            padding: 0; border-radius: 0; font-size: 0.75rem; font-weight: 400;
            margin-right: 8px; margin-bottom: 0;
        }}
        
        .sample-values {{
            display: inline-block;
            max-width: 280px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            background: transparent; padding: 0; border-radius: 0; font-size: 0.75rem;
            color: var(--text-muted); max-height: none;
            border-left: none;
        }}
        
        .sample-values div {{
            padding: 2px 0;
        }}

        .sample-table {{
            font-size: 0.8rem;
        }}

        @media (max-width: 768px) {{
            .modal-content {{
                width: 98%;
                max-height: 92vh;
            }}

            .modal-header {{
                padding: 14px 16px;
            }}

            #modal-body {{
                padding: 16px;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
                gap: 10px;
            }}

            .columns-table th,
            .columns-table td {{
                padding: 8px 10px;
                font-size: 0.8rem;
            }}

            .sample-values {{
                max-width: 180px;
            }}
        }}
    </style>
</head>
<body>
    <div id="app">
        <div class="controls-top">
            <div class="control-group">
                <label>Filtrar Macrotema:</label>
                <input type="text" id="search-macro" placeholder="ex: segurança...">
            </div>
            <div class="control-group">
                <label>Buscar Tema:</label>
                <input type="text" id="search-theme" placeholder="ex: educacao, diabetes...">
            </div>

        </div>

        <div class="main-content">
            <div id="viz-container">
                <svg id="viz"></svg>
                <div id="tooltip"></div>
                <div class="stats-panel">
                    <div class="stat-item">Total de Temas: <span class="stat-value" id="stat-total-themes">0</span></div>
                    <div class="stat-item">Neurônios Ativos: <span class="stat-value" id="stat-active-neurons">0</span></div>
                    <div class="stat-item">Densidade Média: <span class="stat-value" id="stat-avg-density">0</span></div>
                </div>
            </div>
            
            <div id="sidebar">
                <div class="sidebar-empty">
                    Clique em um hexágono para fixar a visualização lateral
                </div>
            </div>
        </div>

        <div id="legend"></div>
    </div>

    <!-- MODAL PARA DETALHES DE TABELA -->
    <div id="modal-overlay" class="modal-overlay">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-table-name">Nome da Tabela</h2>
                <button class="modal-close" onclick="closeTableModal()">&times;</button>
            </div>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        const data = {data_json};
        const macros = {macros_json};
        const metadataDict = {metadata_json};
        const svg = d3.select("#viz");
        const g = svg.append("g");
        const tooltip = d3.select("#tooltip");
        const sidebar = d3.select("#sidebar");

        let macroQuery = "";
        let themeQuery = "";
        let sidebarFilter = "";
        let activeMacro = null;
        let selectedNeuronId = null;

        const xExtent = d3.extent(data, d => d.x);
        const yExtent = d3.extent(data, d => d.y);
        const xRange = xExtent[1] - xExtent[0];
        const maxY = yExtent[1];
        const yRange = maxY - yExtent[0];

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        function getHexPath(radius) {{
            const points = [];
            for (let i = 0; i < 6; i++) {{
                const angle = (Math.PI / 3) * i - (Math.PI / 6);
                points.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
            }}
            return "M" + points.join("L") + "Z";
        }}

        function updateSidebar(d) {{
            if (!d || d.count === 0) {{
                sidebar.html('<div class="sidebar-empty">Clique em um hexágono para fixar a visualização lateral</div>');
                return;
            }}
            
            const tQ = themeQuery.toLowerCase().trim();
            const sF = sidebarFilter.toLowerCase().trim();
            
            const filteredThemes = d.themes.filter(t => 
                (sF === "" || t.theme.toLowerCase().includes(sF))
            );

            let html = `
                <div class="sidebar-header">
                    <h2>${{d.macro || "Sem Macrotema"}}</h2>
                    <div class="neuron-info">Neurônio (${{d.x}}, ${{d.y}}) — ${{d.count}} temas</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 5px;">Distância U-Matrix: ${{d.u_val.toFixed(4)}}</div>
                </div>
                <div class="sidebar-search">
                    <input type="text" id="sidebar-filter-input" placeholder="Filtrar nesta lista..." value="${{sidebarFilter}}" 
                        oninput="sidebarFilter = this.value; updateSidebar(data.find(n => n.id === '${{d.id}}'))">
                </div>
                <div class="sidebar-content" style="flex: 1; overflow-y: auto; padding: 0 20px 20px;">
                    <div style="font-weight: bold; margin: 15px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 5px; font-size: 0.9rem; position: sticky; top: 0; background: var(--sidebar-bg); z-index: 1;">
                        Temas e Tabelas (${{filteredThemes.length}}):
                    </div>
            `;
            
            html += filteredThemes.map(t => {{
                const match = tQ !== "" && t.theme.toLowerCase().includes(tQ);
                const highlightStyle = match ? 'style="background: rgba(255, 159, 67, 0.15); border-left: 3px solid var(--primary); padding-left: 8px;"' : 'style="padding-left: 8px;"';
                
                const tablesHtml = t.tables && t.tables.length > 0 
                    ? `<div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;">
                         ${{t.tables.map(tab => `<span class="table-tag" onclick="openTableModal('${{tab}}')">${{tab}}</span>`).join("")}}
                       </div>`
                    : "";

                return `
                    <div class="theme-list-item" ${{highlightStyle}}>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <span style="font-weight: 600; font-size: 0.85rem;">${{t.theme}}</span>
                            <span class="theme-freq" title="Frequência">${{t.freq}}</span>
                        </div>
                        ${{tablesHtml}}
                    </div>
                `;
            }}).join("");
            
            if (filteredThemes.length === 0) {{
                html += `<div style="text-align: center; color: var(--text-muted); padding: 20px; font-style: italic;">Nenhum tema corresponde ao filtro.</div>`;
            }}

            html += `</div>`;
            sidebar.html(html);
            
            const input = document.getElementById('sidebar-filter-input');
            if (input) {{
                input.focus();
                input.setSelectionRange(input.value.length, input.value.length);
            }}
        }}

        function handleHexClick(event, d) {{
            if (event.button !== 0) return;
            if (d.count === 0) return;
            
            if (selectedNeuronId === d.id) {{
                selectedNeuronId = null;
                sidebarFilter = "";
                updateSidebar(null);
            }} else {{
                selectedNeuronId = d.id;
                sidebarFilter = "";
                updateSidebar(d);
            }}
            
            g.selectAll(".hexagon").classed("selected", d_hex => d_hex.id === selectedNeuronId);
            event.stopPropagation();
        }}

        function render() {{
            const currentTransform = d3.zoomTransform(svg.node());
            g.selectAll("*").remove();
            
            const containerWidth = document.getElementById("viz-container").clientWidth;
            const containerHeight = document.getElementById("viz-container").clientHeight;
            
            const hexW_target = (containerWidth * 0.85) / (xRange + 1);
            const hexH_target = (containerHeight * 0.85) / (yRange + 1);
            
            const radius = Math.min(hexW_target / Math.sqrt(3), hexH_target / 1.5);
            const hexW = Math.sqrt(3) * radius;
            const hexH = 2 * radius;
            const vertDist = (3/4) * hexH;
            const totalHeight = (yRange) * vertDist + hexH;
            
            const marginLeft = 60; 
            const marginTop = 70;
            
            const initialScale = 0.8;
            const transform = d3.zoomIdentity
                .translate(marginLeft, marginTop)
                .scale(initialScale);
            svg.call(zoom.transform, transform);

            for (let i = xExtent[0]; i <= xExtent[1]; i++) {{
                const px = (i - xExtent[0]) * hexW;
                g.append("line").attr("class", "grid-line").attr("x1", px).attr("y1", -20).attr("x2", px).attr("y2", totalHeight);
                g.append("text").attr("class", "axis-label").attr("x", px).attr("y", totalHeight + 20).attr("text-anchor", "middle").text(i);
            }}
            
            for (let i = yExtent[0]; i <= maxY; i++) {{
                const invertedY = maxY - i;
                const py = (invertedY - yExtent[0]) * vertDist;
                g.append("text").attr("class", "axis-label").attr("x", -30).attr("y", py + 4).attr("text-anchor", "end").text(i);
            }}

            const hexes = g.selectAll(".hexagon")
                .data(data)
                .enter()
                .append("path")
                .attr("class", d => {{
                    let classes = d.count > 0 ? "hexagon" : "hexagon empty";
                    if (selectedNeuronId === d.id) classes += " selected";
                    return classes;
                }})
                .attr("d", getHexPath(radius))
                .attr("transform", d => {{
                    const invertedY = maxY - d.y;
                    const px = (d.x - xExtent[0]) * hexW + (invertedY % 2 === 1 ? hexW / 2 : 0);
                    const py = (invertedY - yExtent[0]) * vertDist;
                    return `translate(${{px}}, ${{py}})`;
                }})
                .attr("fill", d => d.color)
                .on("mouseover", function(event, d) {{
                    if(d.count === 0) return;
                    d3.select(this).classed("highlight", true);
                    tooltip.style("display", "block");
                    updateTooltip(event, d);
                    if (selectedNeuronId === null) updateSidebar(d);
                }})
                .on("mousemove", (event) => {{
                    let x = event.clientX + 20;
                    let y = event.clientY + 20;
                    const tooltipWidth = 300;
                    const tooltipHeight = tooltip.node().offsetHeight || 200;
                    if (x + tooltipWidth > window.innerWidth) x = event.clientX - tooltipWidth - 20;
                    if (y + tooltipHeight > window.innerHeight) y = event.clientY - tooltipHeight - 20;
                    tooltip.style("left", x + "px").style("top", y + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this).classed("highlight", false);
                    tooltip.style("display", "none");
                    if (selectedNeuronId === null) updateSidebar(null);
                }})
                .on("click", handleHexClick);

            updateStats();
            applyFilters();
        }}

        function updateTooltip(event, d) {{
            let content = `<span class="t-macro">${{d.macro || "Sem Macrotema"}}</span>`;
            content += `<div style="margin-bottom:8px; font-size:0.7rem; color:#aaa;">Neurônio: (${{d.x}}, ${{d.y}}) | Temas: ${{d.count}}</div>`;
            const tQ = themeQuery.toLowerCase().trim();
            const visibleThemes = d.themes.slice(0, 10);
            content += visibleThemes.map(t => {{
                const match = tQ !== "" && t.theme.toLowerCase().includes(tQ);
                const style = match ? 'style="color:var(--primary); font-weight:bold;"' : '';
                return `<div ${{style}}>• ${{t.theme}} (${{t.freq}})</div>`;
            }}).join("");
            if(d.themes.length > 10) content += `<div style="color:var(--primary); margin-top:4px;">+ ${{d.themes.length - 10}} outros (veja na lateral)</div>`;
            tooltip.html(content);
        }}

        function applyFilters() {{
            const mQ = macroQuery.toLowerCase().trim();
            const tQ = themeQuery.toLowerCase().trim();
            g.selectAll(".hexagon:not(.empty)").each(function(d) {{
                const macroName = (d.macro || "").toLowerCase();
                const hasMacroMatch = mQ === "" || macroName.includes(mQ);
                const isMacroActive = !activeMacro || d.macro === activeMacro;
                const hasThemeMatch = tQ === "" || d.themes.some(t => t.theme.toLowerCase().includes(tQ));
                const isVisible = hasMacroMatch && isMacroActive && hasThemeMatch;
                const isSearching = mQ !== "" || tQ !== "" || activeMacro !== null;
                d3.select(this).classed("dimmed", isSearching && !isVisible).classed("highlight", isSearching && isVisible);
            }});
        }}

        function updateStats() {{
            const activeNeurons = data.filter(d => d.count > 0);
            const totalThemes = d3.sum(activeNeurons, d => d.count);
            const avgDensity = totalThemes / activeNeurons.length || 0;
            
            document.getElementById("stat-total-themes").innerText = totalThemes;
            document.getElementById("stat-active-neurons").innerText = activeNeurons.length;
            document.getElementById("stat-avg-density").innerText = avgDensity.toFixed(1);
        }}

        function buildLegend() {{
            const leg = d3.select("#legend");
            leg.selectAll("*").remove();
            leg.append("div").attr("class", "legend-section-title").text("Macrotemas (ordenados por frequência)");
            const colors = ["#01696f","#EF553B","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52","#636EFA","#7FDBFF","#2ECC40","#FFDC00","#FF851B","#85144b","#3D9970","#a29bfe","#fd79a8","#00b894","#e17055"];
            macros.forEach((m, i) => {{
                const item = leg.append("div").attr("class", "legend-item")
                    .on("click", function() {{
                        const isActive = d3.select(this).classed("active");
                        leg.selectAll(".legend-item").classed("active", false);
                        if(!isActive) {{ d3.select(this).classed("active", true); activeMacro = m; }}
                        else {{ activeMacro = null; }}
                        applyFilters();
                    }});
                item.append("span").attr("class", "legend-dot").style("background", colors[i % colors.length]);
                item.append("span").text(`${{i + 1}} - ${{m}}`);
            }});
        }}

        // ===== FUNÇÕES DO MODAL - VERSÃO COM CONTEÚDO ENRIQUECIDO =====

        function openTableModal(tableName) {{
            // Remover schema do nome da tabela (ex: "public.table_name" -> "table_name")
            const cleanTableName = tableName.includes('.') ? tableName.split('.')[1] : tableName;
            const tableData = metadataDict[cleanTableName];
            if (!tableData) {{
                alert(`Dados da tabela "${{tableName}}" não encontrados.`);
                return;
            }}

            document.getElementById("modal-table-name").textContent = tableName;
            
            let modalBody = `
                <div class="modal-section">
                    <h3>Informações Gerais</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">Schema</div>
                            <div class="info-value">${{tableData.schema || "-"}}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Número de Linhas</div>
                            <div class="info-value">${{tableData.row_count ? tableData.row_count.toLocaleString() : "-"}}</div>
                        </div>
            `;

            if (tableData.primary_key && tableData.primary_key.length > 0) {{
                modalBody += `
                    <div class="info-item">
                        <div class="info-label">Chave Primária</div>
                        <div class="info-value">${{tableData.primary_key.join(", ")}}</div>
                    </div>
                `;
            }}

            if (tableData.foreign_keys && tableData.foreign_keys.length > 0) {{
                modalBody += `
                    <div class="info-item">
                        <div class="info-label">Chaves Estrangeiras</div>
                        <div class="info-value">${{tableData.foreign_keys.length}} encontradas</div>
                    </div>
                `;
            }}

            if (tableData.indexes && tableData.indexes.length > 0) {{
                modalBody += `
                    <div class="info-item">
                        <div class="info-label">Índices</div>
                        <div class="info-value">${{tableData.indexes.length}} encontrados</div>
                    </div>
                `;
            }}

            modalBody += `</div></div>`;

            // Seção de Índices (detalhes)
            if (tableData.indexes && tableData.indexes.length > 0) {{
                const validIndexes = tableData.indexes.filter(idx => idx && idx.name);
                if (validIndexes.length > 0) {{
                    modalBody += `
                        <div class="modal-section">
                            <h3>Detalhes dos Índices</h3>
                            <div class="table-wrapper">
                                <table class="columns-table">
                                    <thead>
                                        <tr>
                                            <th>Nome do Índice</th>
                                            <th>Colunas</th>
                                            <th>Tipo</th>
                                            <th>Único</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    `;
                    validIndexes.forEach(idx => {{
                        const isUnique = idx.is_unique || idx.unique ? "Sim" : "Não";
                        const columns = (idx.columns && Array.isArray(idx.columns)) ? idx.columns.join(", ") : (idx.column_name || "-");
                        const indexType = idx.type || idx.index_type || "-";
                        modalBody += `
                            <tr>
                                <td><strong>${{idx.name}}</strong></td>
                                <td>${{columns}}</td>
                                <td>${{indexType}}</td>
                                <td>${{isUnique}}</td>
                            </tr>
                        `;
                    }});
                    modalBody += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }}
            }}

            // Seção de Colunas
            if (tableData.columns && tableData.columns.length > 0) {{
                modalBody += `
                    <div class="modal-section">
                        <h3>Colunas (${{tableData.columns.length}})</h3>
                        <div class="table-wrapper">
                            <table class="columns-table">
                                <thead>
                                    <tr>
                                        <th>Nome</th>
                                        <th>Tipo</th>
                                        <th>Nullable</th>
                                        <th>Exemplos</th>
                                    </tr>
                                </thead>
                                <tbody>
                `;

                tableData.columns.forEach(col => {{
                    const nullable = col.nullable ? "Sim" : "Não";
                    let examples = "";
                    
                    if (col.stats && col.stats.sample_values && col.stats.sample_values.length > 0) {{
                        examples = col.stats.sample_values
                            .slice(0, 3)
                            .map(v => v !== null && v !== undefined ? String(v) : "NULL")
                            .join(", ");
                        if (col.stats.sample_values.length > 3) {{
                            examples += "...";
                        }}
                    }}

                    modalBody += `
                        <tr>
                            <td><strong>${{col.name}}</strong></td>
                            <td>${{col.type || "—"}}</td>
                            <td>${{nullable}}</td>
                            <td title="${{examples || "—"}}"><span class="sample-values">${{examples || "—"}}</span></td>
                        </tr>
                    `;
                }});

                modalBody += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }}

            // Seção de Estatísticas (se disponível)
            if (tableData.columns && tableData.columns.some(c => c.stats)) {{
                const statsColumns = tableData.columns.filter(c => c.stats && (c.stats.null_count !== undefined || c.stats.unique_count !== undefined || c.stats.min_value !== undefined || c.stats.max_value !== undefined));
                if (statsColumns.length > 0) {{
                    modalBody += `
                        <div class="modal-section">
                            <h3>Estatísticas das Colunas</h3>
                            <div class="table-wrapper">
                                <table class="columns-table">
                                    <thead>
                                        <tr>
                                            <th>Coluna</th>
                                            <th>Valores Nulos</th>
                                            <th>% Nulos</th>
                                            <th>Valores Únicos</th>
                                            <th>Mínimo</th>
                                            <th>Máximo</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    `;
                    statsColumns.forEach(col => {{
                        const nullCount = col.stats.null_count !== undefined ? col.stats.null_count : "-";
                        const nullCountPercentage = (col.stats.null_count !== undefined && tableData.row_count) ? ((col.stats.null_count / tableData.row_count) * 100).toFixed(2) + "%" : "-";
                        const uniqueCount = col.stats.distinct_count !== undefined ? col.stats.distinct_count : "-";
                        const min = col?.stats?.numeric_stats?.min;
                        const max = col?.stats?.numeric_stats?.max;

                        const minValue = min != null ? String(min).substring(0, 50) : "-";
                        const maxValue = max != null ? String(max).substring(0, 50) : "-";
                        modalBody += `
                            <tr>
                                <td><strong>${{col.name}}</strong></td>
                                <td>${{nullCount}}</td>
                                <td>${{nullCountPercentage}}</td>
                                <td>${{uniqueCount}}</td>
                                <td title="${{minValue}}">${{minValue}}</td>
                                <td title="${{maxValue}}">${{maxValue}}</td>
                            </tr>
                        `;
                    }});
                    modalBody += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }}
            }}

            // Seção de Constraints (se disponível)
            if (tableData.constraints && tableData.constraints.length > 0) {{
                const validConstraints = tableData.constraints.filter(c => c && c.name);
                if (validConstraints.length > 0) {{
                    modalBody += `
                        <div class="modal-section">
                            <h3>Constraints</h3>
                            <div class="table-wrapper">
                                <table class="columns-table">
                                    <thead>
                                        <tr>
                                            <th>Nome</th>
                                            <th>Tipo</th>
                                            <th>Definição</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    `;
                    validConstraints.forEach(constraint => {{
                        const constraintType = constraint.type || constraint.constraint_type || "-";
                        const definition = constraint.definition || constraint.constraint_definition || "-";
                        const defStr = String(definition);
                        modalBody += `
                            <tr>
                                <td><strong>${{constraint.name}}</strong></td>
                                <td>${{constraintType}}</td>
                                <td title="${{defStr}}">${{defStr.substring(0, 50)}}${{defStr.length > 50 ? "..." : ""}}</td>
                            </tr>
                        `;
                    }});
                    modalBody += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }}
            }}

            document.getElementById("modal-body").innerHTML = modalBody;
            document.getElementById("modal-overlay").classList.add("active");
        }}

        function closeTableModal() {{
            document.getElementById("modal-overlay").classList.remove("active");
        }}

        // Fechar modal ao clicar fora
        document.getElementById("modal-overlay").addEventListener("click", function(e) {{
            if (e.target === this) {{
                closeTableModal();
            }}
        }});

        // Fechar modal com Escape
        document.addEventListener("keydown", function(e) {{
            if (e.key === "Escape") {{
                closeTableModal();
            }}
        }});

        document.getElementById("search-macro").addEventListener("input", (e) => {{ macroQuery = e.target.value; applyFilters(); }});
        document.getElementById("search-theme").addEventListener("input", (e) => {{ themeQuery = e.target.value; applyFilters(); }});
        window.addEventListener("resize", () => {{ render(); }});
        setTimeout(() => {{ render(); buildLegend(); }}, 100);
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f: f.write(html)

# ======================================================
# MAIN
# ======================================================
def main():
    print("\n=== SOM PIPELINE - COLMEIA HEXAGONAL COM ANOTAÇÕES ===\n")

    api_key = load_env_vars()

    try:
        themes, occurrences, tables_by_theme = load_all_topics()
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    embeddings = get_embeddings(themes, api_key)

    if BEST_PARAMS_FILE.exists():
        print(f"Usando parâmetros otimizados existentes: {BEST_PARAMS_FILE}")
        with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
            best = json.load(f)
    else:
        best = grid_search_som(embeddings)

    print(f"Treinando modelo final com grade {best['x']}x{best['y']}...")
    som = MiniSom(
        best["x"], best["y"], embeddings.shape[1],
        sigma=best["sigma"],
        learning_rate=best["learning_rate"],
        random_seed=SEED,
        topology="hexagonal"
    )

    som.random_weights_init(embeddings)
    som.train_batch(embeddings, best["iterations"], verbose=True)

    # ASSIGNMENTS & METRICS
    clusters = defaultdict(list)
    qe_by_theme = {}
    weights = som.get_weights()

    for i, theme in enumerate(themes):
        neuron = som.winner(embeddings[i])
        clusters[neuron].append(theme)
        qe_by_theme[theme] = float(np.linalg.norm(embeddings[i] - weights[neuron[0], neuron[1]]))

    # MACROTEMAS
    theme_to_idx = {t: i for i, t in enumerate(themes)}
    macrothemes = []

    for neuron, theme_list in clusters.items():
        idxs = [theme_to_idx[t] for t in theme_list]
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_embs)[0]
        rep = theme_list[int(np.argmax(sims))]

        macrothemes.append({
            "macrotema": rep,
            "neuron": f"{neuron[0]}_{neuron[1]}",
            "subtemas": theme_list,
            "frequencia_total": int(sum(occurrences[t] for t in theme_list)),
            "num_subtemas": int(len(theme_list))
        })

    macrothemes.sort(key=lambda x: x["frequencia_total"], reverse=True)

    with open(OUTPUT_DIR / "som_macrothemes.json", "w", encoding="utf-8") as f:
        json.dump(macrothemes, f, indent=2, ensure_ascii=False)

    # OUTLIERS
    threshold = float(np.percentile(list(qe_by_theme.values()), 95))
    outliers = [{"tema": t, "qe": qe} for t, qe in qe_by_theme.items() if qe > threshold]
    outliers.sort(key=lambda x: x["qe"], reverse=True)

    with open(OUTPUT_DIR / "som_outliers.json", "w", encoding="utf-8") as f:
        json.dump(outliers, f, indent=2, ensure_ascii=False)

    # VISUALIZAÇÃO - U-MATRIX COLMEIA COM ANOTAÇÕES
    print("\n Gerando visualização em formato de colmeia com anotações...")
    umatrix = compute_umatrix_hexagonal(som)

    # 1) Figura COLMEIA para o artigo com TOP 10 macrotemas anotados
    plot_umatrix_hex_static(
        umatrix, 
        OUTPUT_DIR / "fig_umatrix_hex_colmeia.png",
        macrothemes=macrothemes,
        top_n=10
    )

    # RELATÓRIOS
    theme_to_neuron = {}
    winners = []

    for i, theme in enumerate(themes):
        neuron = som.winner(embeddings[i])
        winners.append(neuron)
        theme_to_neuron[theme] = f"{neuron[0]}_{neuron[1]}"

    export_macrothemes_table(macrothemes, OUTPUT_DIR / "som_macrothemes_table.csv", top_n=40, max_subthemes=10)
    export_outliers_table(outliers, theme_to_neuron, occurrences, OUTPUT_DIR / "som_outliers_table.csv", top_n=40)
    plot_top_macrothemes_bar(macrothemes, OUTPUT_DIR / "fig_top_macrothemes.png", top_n=20)

    # TABELAS DE TEMAS E TABELAS POR CLUSTER
    print("\nGerando tabelas de temas e tabelas de origem por cluster...")
    export_cluster_theme_tables(
        macrothemes, tables_by_theme, occurrences,
        OUTPUT_DIR / "som_cluster_temas_tabelas.csv"
    )
    export_tables_by_cluster(
        macrothemes, tables_by_theme,
        OUTPUT_DIR / "som_tabelas_por_cluster.csv"
    )

    hits = compute_hits_matrix(best["x"], best["y"], winners)
    plot_hits_heatmap(hits, OUTPUT_DIR / "fig_som_hits.png")
    plot_scatter_winners(best["x"], best["y"], winners, OUTPUT_DIR / "fig_som_scatter_winners.png")

    # CARREGANDO METADATA
    print("\nCarregando metadata das tabelas...")
    metadata_path = DATA_DIR / "step2_output" / "metadata.json"
    metadata_dict = load_metadata_json(metadata_path)
    print(f"Metadata carregada: {len(metadata_dict)} tabelas encontradas")

    # NOVO MAPA INTERATIVO REFINADO COM MODAL
    build_interactive_map_v3(
        som, themes, embeddings, macrothemes, occurrences, tables_by_theme,
        OUTPUT_DIR / "mapa_interativo_refinado.html",
        umatrix,
        metadata_dict=metadata_dict
    )

if __name__ == "__main__":
    main()
