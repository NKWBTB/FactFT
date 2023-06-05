from summac.benchmark import SummaCBenchmark
benchmark_val = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut="val")
frank_dataset = benchmark_val.get_dataset("frank")