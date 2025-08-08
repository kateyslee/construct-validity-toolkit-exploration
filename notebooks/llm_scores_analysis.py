import numpy as np
import pandas as pd
import random

import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def _build_assertion_labels(assertion_dictionary):
    assertion_labels = []
    wrapped_labels = []
    for prop, assertions in assertion_dictionary.items():
        for a_id, assertion_text in assertions.items():
            label = f"{prop.capitalize()} - {a_id}"
            wrapped = textwrap.fill(assertion_text, width=50)
            full_label = f"{label}\n{wrapped}"
            assertion_labels.append(label)
            wrapped_labels.append(full_label)
    return assertion_labels, wrapped_labels

def _property_index_map(assertion_dictionary, properties):
    """Map each property to its global assertion column indices (0..num_assertions-1)."""
    prop_to_inds = {}
    global_index = 0
    for prop in properties:
        if prop in assertion_dictionary:
            prop_to_inds[prop] = []
            for _ in assertion_dictionary[prop].values():
                prop_to_inds[prop].append(global_index)
                global_index += 1
    return prop_to_inds

def analyze_single_model(
    scores,
    assertion_dictionary,
    properties,
    explains=None,
    model_name="(model)",
    plot=True
):
    report = {}

    print(f"\n=== {model_name} ===")

    avg_per_item = scores.mean(axis=2)  # (items, assertions)
    avg_rating_per_assertion = avg_per_item.mean(axis=0)  # (assertions,)

    if not np.all((avg_per_item == 0) | (avg_per_item == 1)):
        print(f"[WARN] {model_name}: avg_per_item contains non-binary values.")

    print("avg_rating_per_assertion:\n", avg_rating_per_assertion)

    #### Assertion-level correlation heatmap
    if plot and assertion_dictionary is not None:
        corr = np.corrcoef(avg_per_item.T)  # (assertions, assertions)
        assertion_keys = [
            assertion_id
            for prop_dict in assertion_dictionary.values()
            for assertion_id in prop_dict.keys()
        ]
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm",
                    xticklabels=assertion_keys, yticklabels=assertion_keys)
        plt.title(f"{model_name} – Assertions Correlation")
        plt.tight_layout()
        plt.show()

    if (assertion_dictionary is not None) and (properties is not None):
        prop_to_inds = _property_index_map(assertion_dictionary, properties)
        property_scores = {
            prop: avg_per_item[:, inds].mean(axis=1)
            for prop, inds in prop_to_inds.items()
        }

        ### Property correlation heatmap
        if plot:
            property_matrix = np.stack(list(property_scores.values()), axis=1)
            property_corr = np.corrcoef(property_matrix.T)
            plt.figure(figsize=(6, 5))
            sns.heatmap(property_corr, annot=True, cmap="coolwarm",
                        xticklabels=property_scores.keys(),
                        yticklabels=property_scores.keys())
            plt.title(f"{model_name} – Property Correlations")
            plt.tight_layout()
            plt.show()

        ### Composite score across all properties
        composite_scores = np.vstack(list(property_scores.values())).mean(axis=0)

        ### Composite histogram
        if plot:
            plt.figure(figsize=(6, 4))
            sns.histplot(composite_scores, bins=10, stat="probability")
            plt.title(f"{model_name} – Composite")
            plt.tight_layout()
            plt.show()

            ### Per-property histograms
            props = list(property_scores.keys())
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            for i, prop in enumerate(props[:4]):  
                sns.histplot(property_scores[prop], bins=5, stat="probability", ax=axes[i])
                axes[i].set_title(f'Property – {prop.capitalize()}')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Proportion')
            plt.tight_layout()
            plt.show()

            ### Bar chart of P(score=1) per assertion
            assertion_labels, wrapped_labels = _build_assertion_labels(assertion_dictionary)
            proportions_1 = [
                np.sum(avg_per_item[:, ind] >= 0.5) / len(avg_per_item[:, ind])
                for ind in range(len(assertion_labels))
            ]
            fig, ax = plt.subplots(figsize=(15, len(wrapped_labels) * 1.2))
            ax.barh(wrapped_labels, proportions_1)
            ax.invert_yaxis()
            ax.set_xlabel('Proportion of Score = 1')
            ax.set_title(f'{model_name} – Assertion Positive Score Proportions (Score = 1 only)')
            ax.set_xlim(0, 1)
            plt.tight_layout()
            plt.show()

    report.update({
        "avg_per_item": avg_per_item,
        "avg_rating_per_assertion": avg_rating_per_assertion,
    })
    return report

def analyze_across_models(score_tables, explain_tables, assertion_dictionary, properties, plot=True):
    import textwrap

    report = {}

    print("\n=== Across Models ===")

    # model_scores: (models, items, assertions)
    model_scores = np.stack([score_tables[m].mean(axis=2) for m in score_tables], axis=0)

    model_order = list(score_tables.keys())
    model_explains = np.stack([explain_tables[m] for m in model_order], axis=0).squeeze()  # (models, ...)

    avg_across_models = model_scores.mean(axis=0)  # (items, assertions)
    avg_rating_per_assertion = avg_across_models.mean(axis=0)  # (assertions,)

    print("avg_rating_per_assertion:\n", avg_rating_per_assertion)

    ### Assertion-level correlation heatmap
    if plot and assertion_dictionary is not None:
        assertion_keys = [
            assertion_id
            for prop_dict in assertion_dictionary.values()
            for assertion_id in prop_dict.keys()
        ]
        corr = np.corrcoef(avg_across_models.T)  # (assertions, assertions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm",
                    xticklabels=assertion_keys, yticklabels=assertion_keys)
        plt.title("Across Models – Assertions Correlation")
        plt.tight_layout()
        plt.show()

    ### Property rollups, correlations, and histograms
    prop_to_inds = _property_index_map(assertion_dictionary, properties)
    property_scores = {
        prop: avg_across_models[:, inds].mean(axis=1)
        for prop, inds in prop_to_inds.items()
    }

    if plot:
        ### Property correlation heatmap
        property_matrix = np.stack(list(property_scores.values()), axis=1)
        property_corr = np.corrcoef(property_matrix.T)
        plt.figure(figsize=(6, 5))
        sns.heatmap(property_corr, annot=True, cmap="coolwarm",
                    xticklabels=property_scores.keys(), yticklabels=property_scores.keys())
        plt.title("Across Models – Property Correlations")
        plt.tight_layout()
        plt.show()
    else:
        property_matrix = np.stack(list(property_scores.values()), axis=1)
        property_corr = np.corrcoef(property_matrix.T)

    ### Composite score across all properties
    composite_scores = np.vstack(list(property_scores.values())).mean(axis=0)

    if plot:
        ### Composite histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(composite_scores, bins=10, stat="probability")
        plt.title("Across Models – Composite")
        plt.tight_layout()
        plt.show()

        ### Per-property histograms
        props_list = list(property_scores.keys())
        n_props = len(props_list)
        nrows = 2 if n_props > 2 else 1
        ncols = 2 if n_props > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for i, prop in enumerate(props_list):
            sns.histplot(property_scores[prop], bins=5, stat="probability", ax=axes[i])
            axes[i].set_title(f'Property – {prop.capitalize()}')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Proportion')

        for j in range(i + 1, len(axes)):  # hide any unused subplots
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    ### Bar chart of P(score=1) per assertion (based on avg_across_models >= 0.5)
    assertion_labels, wrapped_labels = _build_assertion_labels(assertion_dictionary)
    if plot:
        proportions_1 = [
            np.sum(avg_across_models[:, ind] >= 0.5) / len(avg_across_models[:, ind])
            for ind in range(len(assertion_labels))
        ]
        fig, ax = plt.subplots(figsize=(15, len(wrapped_labels) * 1.2))
        ax.barh(wrapped_labels, proportions_1)
        ax.invert_yaxis()
        ax.set_xlabel('Proportion of Score = 1')
        ax.set_title('Across Models – Assertion Positive Score Proportions (Score = 1 only)')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()

    ### Variance across models per assertion
    model_variance = np.var(model_scores, axis=0)  # (items, assertions)
    variance_per_assertion = model_variance.mean(axis=0)  # (assertions,)
    print("variance_per_assertion:\n", variance_per_assertion, variance_per_assertion.shape)

    ### Barh: variance per assertion
    if plot:
        fig, ax = plt.subplots(figsize=(15, len(wrapped_labels) * 1.2))
        ax.barh(wrapped_labels, variance_per_assertion)
        ax.invert_yaxis()
        ax.set_xlabel('Variance Across Models')
        ax.set_title('Across Models – Assertion-Level Variance Across Models')
        ax.set_xlim(0, max(variance_per_assertion) * 1.1 if variance_per_assertion.size else 1)
        plt.tight_layout()
        plt.show()

    ### Average scores per assertion across models (items, assertions)
    avg_scores_per_assertion = model_scores.mean(axis=1)
    print("avg_scores_per_assertion shape:", avg_scores_per_assertion.shape)

    report.update({
        "model_scores": model_scores,
        "model_explains": model_explains,
        "avg_across_models": avg_across_models,
        "avg_rating_per_assertion": avg_rating_per_assertion,

        "property_scores_avg_models": property_scores,
        "property_corr_avg_models": property_corr,
        "composite_scores": composite_scores,

        "model_variance": model_variance,
        "variance_per_assertion": variance_per_assertion,
        "avg_scores_per_assertion": avg_scores_per_assertion,
    })
    return report
