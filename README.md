0. Przekształcenie 3 plików csv do jednego spójnego zbioru danych na dockerze (opcjonalne)
1. Oczyszczenie zbioru danych z brakujących wartości (uzupełnienie średnią dla cech numerycznych, albo najczęstsza wartość dla cech kategorycznych), pozbycie się znaków specjalnych ($)
2. Połączenie labeli z pliku json ze zbiorem danych w celu utworzenia podziału na X i y oraz żeby móc zastosować algorytmy klasteryzacji, regresji itp
3. EDA - plotowanie
4. PCA
5. Zastosowanie KMeans oraz innych algorytmów klastrujących
6. Zastosowanie IsolationForest do detekcji anomalii
7. Ciąg dalszy nastąpi...


Kod do *pierwszego* klonowania repo:

Tworzy skrypt, który jest wykonywany w trakcie `git commit`

Zwalnia z potrzeby ręcznego formatowania kodu przez `ruff`

```sh
git clone https://github.com/rrubaszek/TransactionDatasetAnalysis.git && \
cd TransactionDatasetAnalysis && \
cat > .git/hooks/pre-commit <<'EOF'
#!/usr/bin/env sh
set -e

STAGED_PYTHON_FILES=".git/.staged-python-files"
git diff --cached --name-only --diff-filter=ACMR -z -- '*.py' '*.pyi' '*.ipynb' > "$STAGED_PYTHON_FILES"

if [ -s "$STAGED_PYTHON_FILES" ]; then
  xargs -0 uv run ruff check --fix < "$STAGED_PYTHON_FILES"
  xargs -0 uv run ruff format < "$STAGED_PYTHON_FILES"
  xargs -0 git add -- < "$STAGED_PYTHON_FILES"
fi

rm -f "$STAGED_PYTHON_FILES"
EOF
chmod +x .git/hooks/pre-commit
```
