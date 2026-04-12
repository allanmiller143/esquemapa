# extract_metadata_advanced.py
import os
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import urllib.parse
import json
import re # NOVO: Para análise de padrões
from pathlib import Path
# --- Configurações e Conexão 
load_dotenv()
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5433")
PG_DB   = os.getenv("PG_DB")
PG_PASS_ENC = urllib.parse.quote_plus(PG_PASS)
conn_string = f"postgresql://{PG_USER}:{PG_PASS_ENC}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# --- Flags de Controle ---

GET_COLUMN_STATS = True     
TOP_N_FREQUENT_VALUES = 5    
SAMPLE_ROWS = 3             
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "step1_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funções de Conexão e Listagem de Tabelas ---
def connect(engine_str):
    try:
        engine = create_engine(engine_str)
        conn = engine.connect()
        return engine, conn
    except SQLAlchemyError as e:
        print("Erro ao conectar:", e)
        raise

def list_tables(inspector):
    tables = []
    for schema in inspector.get_schema_names():
        if schema in ("pg_catalog", "information_schema"):
            continue
        for tbl in inspector.get_table_names(schema=schema):
            tables.append((schema, tbl))
    return tables

# Função para extrair estatísticas avançadas de uma coluna
def get_column_advanced_stats(conn, schema, table, column_info, row_count):
    col_name = column_info["name"]
    col_type_str = str(column_info["type"]).upper()
    stats = {}

    # Ignorar colunas com muitos nulos ou tabelas vazias para otimizar
    if row_count == 0:
        return {"error": "Tabela vazia"}

    try:
        # 1. Contagem de Nulos e Distintos
        q_stats = text(f"""
            SELECT
                COUNT(*) FILTER (WHERE "{col_name}" IS NULL) AS null_count,
                COUNT(DISTINCT "{col_name}") AS distinct_count
            FROM "{schema}"."{table}"
        """)
        result = conn.execute(q_stats).first()
        stats['null_count'] = int(result.null_count)
        stats['distinct_count'] = int(result.distinct_count)
        stats['null_percentage'] = round((stats['null_count'] / row_count) * 100, 2) if row_count > 0 else 0

        # 2. Estatísticas para colunas numéricas
        if any(t in col_type_str for t in ['INT', 'BIGINT', 'NUMERIC', 'DECIMAL', 'REAL', 'DOUBLE']):
            q_numeric = text(f"""
                SELECT
                    AVG("{col_name}") AS avg_val,
                    MIN("{col_name}") AS min_val,
                    MAX("{col_name}") AS max_val,
                    STDDEV("{col_name}") AS stddev_val
                FROM "{schema}"."{table}"
            """)
            num_result = conn.execute(q_numeric).first()
            stats['numeric_stats'] = {
                'avg': float(num_result.avg_val) if num_result.avg_val is not None else None,
                'min': float(num_result.min_val) if num_result.min_val is not None else None,
                'max': float(num_result.max_val) if num_result.max_val is not None else None,
                'stddev': float(num_result.stddev_val) if num_result.stddev_val is not None else None,
            }

        # 3. Valores mais frequentes para colunas de texto/categoria
        if TOP_N_FREQUENT_VALUES > 0 and stats.get('distinct_count', 0) < (row_count * 0.5):
             q_freq = text(f"""
                SELECT "{col_name}", COUNT(*) as frequency
                FROM "{schema}"."{table}"
                WHERE "{col_name}" IS NOT NULL
                GROUP BY "{col_name}"
                ORDER BY frequency DESC
                LIMIT {TOP_N_FREQUENT_VALUES}
            """)
             freq_result = conn.execute(q_freq).fetchall()
             stats['frequent_values'] = [{'value': str(row[0]), 'count': int(row[1])} for row in freq_result]

        # 4. 15 primeiros valores não-nulos (exemplos reais)
        # Busca os primeiros 15 valores distintos que não são NULL
        q_sample = text(f"""
            SELECT DISTINCT "{col_name}"
            FROM "{schema}"."{table}"
            WHERE "{col_name}" IS NOT NULL
            LIMIT 15
        """)
        sample_result = conn.execute(q_sample).fetchall()
        stats['sample_values'] = [str(row[0]) for row in sample_result]

    except Exception as e:
        # Usar regex para simplificar a mensagem de erro, que pode ser longa
        error_message = str(e).split('\n')[0]
        if len(error_message) > 150:
             error_message = error_message[:150] + "..."
        stats['error'] = f"Falha ao analisar coluna: {error_message}"
        # print(f"  - Aviso: {stats['error']} em {schema}.{table}.{col_name}")

    return stats


def extract_table_metadata(inspector, conn, schema, table):
    meta = {"schema": schema, "table_name": table}
    fullname = f"{schema}.{table}"

    # Contagem de linhas (feito primeiro para otimizar o resto)
    row_count = 0
    try:
        q_count = text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
        row_count = conn.execute(q_count).scalar()
    except Exception as e:
        print(f"Erro COUNT(*) em {fullname}: {e}")
        row_count = -1
    meta["row_count"] = row_count

    # Metadados básicos (colunas, PK, FK)
    cols_info = inspector.get_columns(table, schema=schema)
    meta["primary_key"] = inspector.get_pk_constraint(table, schema=schema).get("constrained_columns", [])
    meta["foreign_keys"] = inspector.get_foreign_keys(table, schema=schema)

    print(f"  -> {fullname}: {row_count} linhas - {len(cols_info)} colunas encontradas")
    
    # Loop principal para colunas
    processed_columns = []
    for i, col_info in enumerate(cols_info):
        col_data = {
            "name": col_info["name"],
            "type": str(col_info["type"]),
            "nullable": col_info.get("nullable", True)
        }

        print(f"    - Processando coluna {i+1}/{len(cols_info)}: {col_data['name']} ({col_data['type']})")
        # Adiciona estatísticas avançadas se a flag estiver ligada
        if GET_COLUMN_STATS and row_count > 0 :
            advanced_stats = get_column_advanced_stats(conn, schema, table, col_info, row_count)
            col_data['stats'] = advanced_stats
        
        processed_columns.append(col_data)
    
    meta["columns"] = processed_columns

    # Amostra de linhas
    if SAMPLE_ROWS > 0 and row_count > 0:
        try:
            qs = text(f'SELECT * FROM "{schema}"."{table}" LIMIT :lim')
            sample_df = pd.read_sql(qs, conn, params={"lim": SAMPLE_ROWS})
            meta["sample_rows"] = sample_df.to_dict(orient='records')
        except Exception as e:
            meta["sample_rows"] = [{"error": f"Erro ao obter amostra: {e}"}]
    else:
        meta["sample_rows"] = []

    return meta

def main():
    engine, conn = connect(conn_string)
    inspector = inspect(engine)
    tables = list_tables(inspector)
    total_tables = len(tables)
    print(f"PROGRESS_TOTAL:{total_tables}") # Envia o total para o Streamlit

    all_metadata = []
    for i, (schema, table) in enumerate(tables):
        current_idx = i + 1
        print(f"PROGRESS_CURRENT:{current_idx}") # Envia o índice atual
        try:
            # Usar uma transação por tabela para garantir consistência
            with conn.begin():
                m = extract_table_metadata(inspector, conn, schema, table)
                all_metadata.append(m)
        except Exception as e:
            print(f"  -> ERRO FATAL ao processar a tabela {schema}.{table}: {e}")
            # Adiciona um registro de erro para não perder o rastro
            all_metadata.append({"schema": schema, "table_name": table, "error": str(e)})

    print("\nProcessamento concluído. Salvando arquivos consolidados...")
    
    # Salvar o arquivo JSON completo
    json_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # Usar um conversor padrão para lidar com tipos de dados não serializáveis (ex: Decimal)
        json.dump(all_metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metadados avançados salvos em: {json_path}")

    # Salvar o DataFrame de resumo
    summary_list = [{
        "schema": m.get("schema"),
        "table": m.get("table_name"),
        "row_count": m.get("row_count"),
        "column_count": len(m.get("columns", [])),
        "pk": "|".join(m.get("primary_key", []))
    } for m in all_metadata if 'error' not in m]
    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(OUTPUT_DIR, "tables_summary_advanced.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Resumo avançado salvo em: {summary_path}")

    conn.close()
    engine.dispose()

if __name__ == "__main__":
    main()