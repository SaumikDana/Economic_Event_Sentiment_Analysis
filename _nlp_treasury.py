import pandas as pd
import matplotlib.pyplot as plt
import traceback
import json
from datetime import datetime

from _embeddings import (
    create_document_embeddings, 
    calculate_sentiment_scores,
    create_policy_axis,
    integrate_new_labeled_examples,
    scrape_text
)


def plot_sentiment(df):
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    colors = {'contractionary': 'red', 'neutral': 'gray', 'expansionary': 'green'}
    
    for i, (date, sentiment) in enumerate(zip(df_sorted.index, df_sorted['stance'])):
        plt.scatter(date, sentiment, c=colors[sentiment], alpha=0.7, s=10)
    
    for sentiment, color in colors.items():
        plt.scatter([], [], c=color, label=sentiment)
    
    plt.title('Treasury Sentiment Over Time')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_sentiment_score(df, threshold):
    df_sorted = df.sort_index()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    
    plt.figure(figsize=(14, 9))
    
    plt.plot(df_sorted.index, df_sorted['sentiment_score'], 'b.-', linewidth=1, alpha=0.7)
    plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=-threshold, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.title('Treasury Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_stance_label(score, threshold):
    """Convert score to stance"""
    if score > threshold:
        return 'expansionary'
    elif score < -threshold:
        return 'contractionary'
    else:
        return 'neutral'

def analyze_treasury_statements(
    statements_df,
    labeled_examples,
    threshold,
    learn_threshold,   # NEW: stricter gate for self-training
    learn=True,
):
    if statements_df.empty:
        return pd.DataFrame()

    if labeled_examples is None:
        raise ValueError("Must provide labeled_examples with 'expansionary' and 'contractionary' keys")

    if 'expansionary' not in labeled_examples or 'contractionary' not in labeled_examples:
        raise ValueError("labeled_examples must have 'expansionary' and 'contractionary' keys")

    # Default: learn at same strictness as classification unless specified
    if learn_threshold is None:
        learn_threshold = threshold

    print(f"Analyzing {len(statements_df)} Treasury statements...")

    statements_df = statements_df.reset_index(drop=True)

    # Extract valid texts
    texts = []
    valid_indices = []
    for i, row in statements_df.iterrows():
        text = row.get('statement_text', '')
        if isinstance(text, str) and len(text.strip()) > 50:
            texts.append(text)
            valid_indices.append(i)

    if not texts:
        print("❌ No valid text content found")
        return pd.DataFrame()

    try:
        # 1) Embed docs ONCE
        print("Embed Docs ...")
        document_embeddings, model, processed_texts = create_document_embeddings(texts)

        # 2) Build initial axis from current labeled examples
        print("Build Axis ...")
        axis = create_policy_axis(
            model,
            labeled_examples['expansionary'],
            labeled_examples['contractionary']
        )

        # 3) First scoring pass
        print("Get Scores ...")
        scores_v1 = calculate_sentiment_scores(document_embeddings, axis)

        # 4) First pass results + learning
        results = []
        learned_any = False

        for i in range(min(len(scores_v1), len(valid_indices))):
            valid_idx = valid_indices[i]
            if valid_idx >= len(statements_df):
                continue

            row = statements_df.iloc[valid_idx]
            score = float(scores_v1[i])

            # Classification stance (for output)
            stance = get_stance_label(score, threshold)

            date = row.get('date')
            text = row.get('statement_text', '')

            # Learning stance (stricter gate)
            if learn:
                learn_stance = get_stance_label(score, learn_threshold)
                if learn_stance in ("expansionary", "contractionary"):
                    before = len(labeled_examples[learn_stance])

                    labeled_examples = integrate_new_labeled_examples(
                        labeled_examples=labeled_examples,
                        new_text=text,
                        stance=learn_stance,
                        model=model,
                    )

                    after = len(labeled_examples[learn_stance])
                    if after > before:
                        print(f"Added {learn_stance} example {date} (now {after})")
                        learned_any = True

            # store v1 temporarily (will be overwritten if we do pass 2)
            results.append({
                'date': date,
                'sentiment_score': score,
                'stance': stance
            })

        # 5) If learning changed the example bank, rebuild axis ONCE and rescore
        if learn and learned_any:
            print("Update axis...")
            axis = create_policy_axis(
                model,
                labeled_examples['expansionary'],
                labeled_examples['contractionary']
            )
            scores_v2 = calculate_sentiment_scores(document_embeddings, axis)

            # overwrite scores/stances with v2 using same ordering
            for j in range(min(len(scores_v2), len(results))):
                score2 = float(scores_v2[j])
                results[j]['sentiment_score'] = score2
                results[j]['stance'] = get_stance_label(score2, threshold)

        results_df = pd.DataFrame(results)

        print(f"\n✓ Analysis complete (threshold={threshold}, learn_threshold={learn_threshold}):")
        print(f"  Expansionary: {(results_df['stance'] == 'expansionary').sum()}")
        print(f"  Neutral: {(results_df['stance'] == 'neutral').sum()}")
        print(f"  Contractionary: {(results_df['stance'] == 'contractionary').sum()}")

        if learn:
            print("\n✓ Updated labeled_examples (after pruning):")
            print(f"  expansionary: {len(labeled_examples.get('expansionary', []))}")
            print(f"  contractionary: {len(labeled_examples.get('contractionary', []))}")

        return results_df, labeled_examples

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return pd.DataFrame(), {}



def analyze_all_statements(statements_df, labeled_examples, threshold, learn_threshold, plot=False):
    """Main entry point for sentiment analysis"""

    print("\n" + "="*80)
    print("TREASURY FISCAL SENTIMENT ANALYSIS")
    print("="*80)

    results_df, labeled_examples = analyze_treasury_statements(statements_df, labeled_examples, threshold, learn_threshold)

    if not results_df.empty:
        results_df.set_index('date', inplace=True)

        if plot:
            plot_sentiment(results_df)
            plot_sentiment_score(results_df, threshold)

    return results_df, labeled_examples


def treasury_press_release_scraper(start_date, end_date, step=1):
    
    secretary_eras = [
        ("sm", 1, 1240),
        ("jy", 1, 2800),
        ("sb", 1, 400),
    ]

    statements_data = []

    for prefix, num_start, num_end in secretary_eras:
        for num in range(num_start, num_end + 1, step):
            text_content, release_date = scrape_text(prefix, num)
            if text_content is not None and release_date is not None:
                print(f'Scraped: {release_date}')
                statements_data.append({'date': release_date, 'statement_text': text_content})

    all_statements = pd.DataFrame(statements_data)
    all_statements['date'] = pd.to_datetime(all_statements['date'])
    all_statements = all_statements.sort_values('date')
    df_combined = all_statements.groupby('date')['statement_text'].apply(lambda x: '\n'.join(x)).reset_index()
    filtered_df = df_combined[(df_combined['date'] >= start_date) & (df_combined['date'] <= end_date)].copy()

    return filtered_df


def run_treasury_fiscal_analysis(start_date, end_date, labeled_examples, step=1, threshold=1.0, learn_threshold=1.5, plot=False):

    statements_df = treasury_press_release_scraper(start_date, end_date, step=step)
    sentiment_df, labeled_examples = analyze_all_statements(statements_df, labeled_examples, threshold, learn_threshold, plot=plot)

    return sentiment_df, labeled_examples


if __name__ == "__main__":
    

    labeled_examples = {
        "expansionary": [
            # Spending / stimulus / support
            "We will increase federal investment to accelerate economic growth, expand infrastructure spending, and support job creation.",
            "The Administration will provide additional fiscal support to strengthen demand, boost employment, and speed the recovery.",
            "We are committed to expanding targeted relief for households and small businesses to sustain consumption and prevent layoffs.",
            "We will use federal resources to stabilize the economy by increasing spending where it has the highest multiplier effects.",
            "This package increases federal outlays to support aggregate demand and reduce economic slack.",

            # Tax cuts / credits / rebates
            "We will cut taxes for middle-class families to raise disposable income and strengthen consumer spending.",
            "We will expand refundable tax credits to deliver immediate support to working families and increase purchasing power.",
            "We are proposing tax relief that will increase after-tax income and encourage near-term spending and investment.",
            "The plan accelerates tax refunds and provides temporary tax reductions to support demand.",
            "We will implement temporary tax incentives to increase private investment and expand hiring.",

            # Deficit tolerance when economy is weak
            "In the near term, deficit-financed support is appropriate to strengthen the economy and reduce unemployment.",
            "We will prioritize growth and jobs even if it requires higher short-term deficits to avoid a deeper downturn.",
            "Fiscal policy should lean against the downturn with temporary deficit spending to support recovery.",

            # Countercyclical framing
            "We will deploy countercyclical fiscal measures to cushion the economy and support a faster return to full employment.",
            "Additional public investment now will raise output and employment while interest rates remain low.",
        ],

        "contractionary": [
            # Deficit / debt reduction
            "We are committed to reducing the deficit and putting the debt on a sustainable path through spending restraint and reforms.",
            "Fiscal consolidation is necessary to strengthen long-run sustainability by lowering deficits and slowing debt growth.",
            "We will implement measures to bring spending in line with revenues and reduce borrowing needs.",
            "We must take steps to reduce the federal deficit by restraining outlays and improving budget discipline.",
            "The plan reduces projected deficits through a combination of spending cuts and revenue measures.",

            # Spending cuts / caps / sequestration-like language
            "We will reduce federal spending growth by enforcing budget caps and limiting discretionary outlays.",
            "This agreement restrains discretionary spending to achieve meaningful deficit reduction over the next decade.",
            "We are proposing targeted spending reductions to improve fiscal sustainability and reduce the size of government borrowing.",
            "We will phase down temporary programs and reduce federal expenditures as the economy strengthens.",

            # Tax increases / base broadening / enforcement framed as deficit reduction
            "We will raise revenues to reduce deficits by closing loopholes and broadening the tax base.",
            "Revenue measures are necessary to reduce the deficit and stabilize debt as a share of the economy.",
            "We will increase compliance and enforcement to raise revenues and support deficit reduction goals.",

            # Austerity / tightening framing
            "With the economy expanding, we should pivot toward deficit reduction and rebuild fiscal space for future shocks.",
            "We must take difficult steps now to restrain spending and reduce deficits to support long-run growth.",
            "Restoring fiscal sustainability requires near-term restraint and structural reforms to entitlement spending.",
        ],
    }


    sentiment_df, labeled_examples = run_treasury_fiscal_analysis(
        start_date='2017-01-01',
        end_date='2025-12-31',
        labeled_examples=labeled_examples,
        step=20,
        threshold=1.0,
        learn_threshold=1.5,
        plot=True
    )

    # --- Save final labeled examples to JSON ---
    ts = datetime.now().strftime("%Y%m%d")
    out_path = f"labeled_examples_final_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled_examples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved final labeled_examples to: {out_path}")
