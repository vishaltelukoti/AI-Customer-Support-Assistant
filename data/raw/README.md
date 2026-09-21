# Raw dataset provenance

The only selected preprocessing input is `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, supplied locally by the project owner. It is the Customer IT Support - Ticket Dataset by Tobias Bueck, from [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets). The [creator describes it as synthetic](https://softoft.de/blog/ticket-dataset/). It is not production customer data.

The file has 28,587 rows, 16 columns, English/German tickets and three row-version labels. Its raw bytes are preserved; preprocessing filters English rows without deleting German rows. No other dataset variants are downloaded, added or combined.

`sample_tickets.csv` is the original 20-record hand-written Day 1 fixture, retained unchanged solely for historical reference. It is not read by the current pipeline or tests. The application API reads neither CSV.

See [the dataset audit](../../docs/dataset.md) for the raw checksum, field mapping, measurements, privacy handling and split decisions, and [the README](../../README.md) for the exact preprocessing command.
