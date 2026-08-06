"""Entry point pipeline eksperimen faktorial BI-Rate.

Contoh: python run_experiments.py --pipeline
        python run_experiments.py --screening / --analyze / --focused --top 10
        python run_experiments.py --tuning --top 3 / --final / --report / --list
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="BI-Rate Prediction — Factorial Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pipeline", action="store_true",
                        help="Full pipeline (screening → analysis → focused → tuning → final)")
    group.add_argument("--screening", action="store_true",
                        help="Phase 1: Screening (all factorial combinations)")
    group.add_argument("--analyze", action="store_true",
                        help="Phase 2: Factor analysis (requires screening results)")
    group.add_argument("--focused", action="store_true",
                        help="Phase 3: Focused evaluation (top-N with all models)")
    group.add_argument("--tuning", action="store_true",
                        help="Phase 4: Hyperparameter tuning (top-N)")
    group.add_argument("--final", action="store_true",
                        help="Phase 5: Final comparison")
    group.add_argument("--report", action="store_true",
                        help="Generate PDF experiment report")
    group.add_argument("--list", action="store_true",
                        help="Show experiment space and design")

    parser.add_argument("--top", type=int, default=6,
                        help="Top-N combinations for focused/tuning (default: 6)")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip LSTM/BiLSTM")
    parser.add_argument("--output-dir", type=str, default="results/experiments",
                        help="Output directory (default: results/experiments)")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n")
        from src.experiments.scenario_config import list_all_experiments
        list_all_experiments()
        return

    if args.list:
        from src.experiments.scenario_config import list_all_experiments
        list_all_experiments()
        return

    if args.report:
        from src.experiments.experiment_report_generator import generate_experiment_report
        generate_experiment_report(args.output_dir)
        return


    from src.experiments.experiment_runner import FactorialExperimentRunner
    runner = FactorialExperimentRunner(output_dir=args.output_dir)

    if args.pipeline:
        runner.run_pipeline(
            top_focused=args.top,
            top_tuning=max(3, args.top // 3),
            skip_dl=args.skip_dl,
        )
        # Auto-generate report
        try:
            from src.experiments.experiment_report_generator import generate_experiment_report
            generate_experiment_report(args.output_dir)
        except Exception as e:
            print(f"Gagal generate report: {e}")
    elif args.screening:
        runner.run_screening()
    elif args.analyze:
        runner.analyze_factors()
    elif args.focused:
        runner.run_focused(top_n=args.top, skip_dl=args.skip_dl)
    elif args.tuning:
        runner.run_tuning(top_n=args.top)
    elif args.final:
        runner.run_final()

    print(f"\nResults: {args.output_dir}/")


if __name__ == "__main__":
    main()
