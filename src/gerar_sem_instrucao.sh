#!/bin/bash
# Gera as colecoes de embedding SEM a instrucao em portugues, para isolar o
# efeito dela. Rode no Mac (precisa do modelo baixado), a partir da raiz:
#
#     bash src/gerar_sem_instrucao.sh
#
# Depois, avaliar_query.py acrescenta sozinho a linha "gemma_si" na tabela.
# Blocos ja gerados sao pulados: da para interromper e retomar.
set -e
cd "$(dirname "$0")/.."

python3 src/gerar_embeddings_gemma300_benchmark.py --kind queries \
    --no-instruction \
    --output-dir embeddings/gemma300_queries_fase2_sem_instrucao

for v in tr ipc_grupo ipc_direto ipc_hierarquia; do
    case $v in
        tr)             saida=gemma300_tr_docs ;;
        ipc_grupo)      saida=gemma300_tr_ipc_grupo_pt_docs ;;
        ipc_direto)     saida=gemma300_tr_ipc_direto_pt_docs ;;
        ipc_hierarquia) saida=gemma300_tr_ipc_hierarquia_pt_docs ;;
    esac
    echo
    echo "=== $v -> ${saida}_sem_instrucao"
    python3 src/gerar_embeddings_gemma300_benchmark.py --kind docs --variant "$v" \
        --no-instruction \
        --output-dir "embeddings/${saida}_sem_instrucao"
done

echo
echo "Pronto. Agora:  python3 src/avaliar_query.py dados/gabaritos/QN003.tsv"
