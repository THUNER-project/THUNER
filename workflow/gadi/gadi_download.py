
def download_data():
    # Load list of urls from file
    urls = []
    with open("extracted_urls.txt", "r") as f:
        urls = f.readlines()

    parent_local = "/scratch/w40/esh563/THOR_output/input_data/raw"
    parent_remote = "https://data.rda.ucar.edu"
    args_dict = parent_local, parent_remote

    # Download data
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for url in urls:
            time.sleep(1)
            futures.append(executor.submit(data.utils.download, url, **args_dict))
        parallel.check_futures(futures)

