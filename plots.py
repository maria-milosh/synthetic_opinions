import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Plot 1: Histogram of pairwise distances ----------
def plot_pairwise_distance_histogram(pairwise_dists: np.ndarray, output_dir: str, embedding_type: str):
    base = os.path.join(output_dir, f"{embedding_type}_distance_hist")
    plt.figure()
    plt.hist(pairwise_dists, bins=30)
    plt.title("Pairwise cosine distances (all pairs)")
    plt.xlabel("cosine distance")
    plt.ylabel("count")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()
    # plt.show()

# ---------- Plot 2: Heatmap of distances (sorted by persona_id) ----------
def plot_distance_heatmap(D: np.ndarray, df: pd.DataFrame, out_dir: str, embedding_type: str):
    order = np.argsort(df["persona_id"].astype(str).to_numpy())
    D_sorted = D[order][:, order]
    labels_sorted = df["persona_id"].astype(str).to_numpy()[order]

    base = os.path.join(out_dir, f"{embedding_type}_distance_heatmap")
    plt.figure()
    plt.imshow(D_sorted, aspect="auto")
    plt.title("Cosine distance heatmap (sorted by persona_id)")
    plt.xlabel("items (sorted)")
    plt.ylabel("items (sorted)")
    plt.colorbar(label="cosine distance")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()
    # plt.show()

# ---------- Plot 3: Persona-grouped distances ----------
def plot_persona_grouped_distances(df: pd.DataFrame, triu_i: np.ndarray, triu_j: np.ndarray,
                                   pairwise_dists: np.ndarray, out_dir: str, embedding_type: str):
    # Build a long table of pairwise distances with persona labels
    pid = df["persona_id"].astype(str).to_numpy()
    pairs = pd.DataFrame({
        "i": triu_i,
        "j": triu_j,
        "dist": pairwise_dists,
        "persona_i": pid[triu_i],
        "persona_j": pid[triu_j],
    })
    pairs["within"] = (pairs["persona_i"] == pairs["persona_j"])

    # 3a) boxplot differences between persons
    within = pairs.loc[pairs["within"], "dist"].to_numpy()
    between = pairs.loc[~pairs["within"], "dist"].to_numpy()

    base = os.path.join(out_dir, f"{embedding_type}_distance_boxplot")
    plt.figure()
    plt.boxplot([within, between], labels=["within persona", "between personas"])
    # plt.boxplot(between, labels=["between personas"])
    plt.title("Cosine distances")
    plt.ylabel("cosine distance")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()

    # 3b) between-persona pair distributions (A-B same as B-A)
    # This one can get very busy with many personas, so commented out for now.
    # a = np.minimum(pairs["persona_i"], pairs["persona_j"])
    # b = np.maximum(pairs["persona_i"], pairs["persona_j"])
    # pairs["pair"] = a + " vs " + b
    # # pairs["pair"] = pairs["persona_i"] + " vs " + pairs["persona_j"]

    # # Keep only between-persona pairs for this plot
    # between_pairs = pairs.loc[~pairs["within"]].copy()

    # # Sort pairs by median distance for readability
    # pair_order = (
    #     pairs.groupby("pair")["dist"]
    #     .median()
    #     .sort_values()
    #     .index
    #     .tolist())

    # data = [between_pairs.loc[between_pairs["pair"] == p, "dist"].to_numpy() for p in pair_order]

    # base = os.path.join(out_dir, f"{embedding_type}_distance_between_persona_boxplots")

    # plt.figure(figsize=(max(8, len(pair_order) * 0.8), 5))
    # plt.boxplot(data, labels=pair_order, vert=True)
    # plt.title("Cosine distances by persona pair (between only)")
    # plt.ylabel("cosine distance")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    # plt.savefig(base + ".pdf", bbox_inches="tight")
    # plt.close()
    # plt.show()




# ---------- Cosine distance matrix ----------
def compute_pairwise_distances(X: np.ndarray):
    # Given embeddings matrix X (n x d), compute pairwise cosine distance matrix D (n x n).
    # Based on cosine_similarity = (X @ X.T) / (||X|| ||X||)

    # Calculate L2 norms:
    # actually not needed in our pipeline since SentenceTransformer gives normalized embeddings.
    # But keeping it here for completeness.
    # norms = np.linalg.norm(X, axis=1, keepdims=True)
    # norms[norms == 0] = 1.0
    # #checks
    # print("min norm:", norms.min())
    # print("max norm:", norms.max())
    # print("mean norm:", norms.mean())

    # Normalize embeddings
    # Xn = X / norms
    Xn = X  # assuming embeddings are already normalized (check analysis.py - they should be)

    S = Xn @ Xn.T  # cosine similarity
    D = 1.0 - S    # cosine distance; now different they are from 1, where 1 is identical stance

    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 embeddings to compute pairwise distances.")

    # Upper triangle distances (exclude diagonal)
    triu_i, triu_j = np.triu_indices(n, k=1)
    pairwise_dists = D[triu_i, triu_j]

    return D, triu_i, triu_j, pairwise_dists



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--text-field", default="argument_snippet") # or "justification_bullets"
    args = ap.parse_args()

    output_dir = os.path.join("outputs", args.input_folder, "similarity_plots")
    os.makedirs(output_dir, exist_ok=True)
    EMB_PATH = os.path.join("outputs", args.input_folder, f"embeddings_{args.text_field}.jsonl")

    rows = []
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Required columns
    if "embedding" not in df.columns:
        raise ValueError("Missing 'embedding' column in embeddings file.")

    # Optional columns
    if "record_id" not in df.columns:
        df["record_id"] = [f"row_{i:06d}" for i in range(len(df))]
    if "persona_id" not in df.columns:
        df["persona_id"] = "unknown"

    X = np.vstack(df["embedding"].to_list()).astype(float)

    D, triu_i, triu_j, pairwise_dists = compute_pairwise_distances(X)

    plot_pairwise_distance_histogram(pairwise_dists, output_dir, args.text_field)
    plot_distance_heatmap(D, df, output_dir, args.text_field)
    plot_persona_grouped_distances(df, triu_i, triu_j, pairwise_dists, output_dir, args.text_field)

    print(f"{args.text_field} plots were saved to: {output_dir}")



if __name__ == "__main__":
    main()