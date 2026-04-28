from transaction_analysis.data import bootstrap, cleanup


def main() -> None:
    bootstrap.run(force=True)
    cleanup.run(
        force=True,
    )


if __name__ == "__main__":
    main()
