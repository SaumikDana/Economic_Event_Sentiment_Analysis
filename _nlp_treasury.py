import pandas as pd
import json
from datetime import datetime
from _embeddings import scrape_text, analyze_treasury_statements, plot_sentiment, plot_sentiment_score



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

    start_date='2017-01-01'
    end_date='2025-12-31'
    step=5
    threshold=1.0
    learn_threshold=1.5
    plot=True

    statements_df = treasury_press_release_scraper(start_date, end_date, step=step)
    sentiment_df, labeled_examples = analyze_all_statements(statements_df, labeled_examples, threshold, learn_threshold, plot=plot)

    # --- Save final labeled examples to JSON ---
    ts = datetime.now().strftime("%Y%m%d")
    out_path = f"labeled_examples_final_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled_examples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved final labeled_examples to: {out_path}")
