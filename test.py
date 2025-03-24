def process_file(file_path, processed_directory):
    try:
        with gzip.open(file_path, 'rb') as f:
            file_content = f.read()
            articles = parse_xml_content(file_content)

            # Prepare titles and abstracts for bulk encoding
            titles = [article["title"] for article in articles if article["title"]]
            abstracts = [article["abstract"] for article in articles if article["abstract"]]

            # Encode titles and abstracts in bulk
            title_vectors = encode_texts(titles) if titles else []
            abstract_vectors = encode_texts(abstracts) if abstracts else []

            # Ensure the lengths match
            if len(title_vectors) != len(titles):
                raise ValueError("Mismatch between number of titles and title vectors.")
            if len(abstract_vectors) != len(abstracts):
                raise ValueError("Mismatch between number of abstracts and abstract vectors.")

            # Assign vectors back to articles
            title_idx = 0
            abstract_idx = 0
            for article in articles:
                if article["title"]:
                    if title_idx < len(title_vectors):
                        article["title_vector"] = title_vectors[title_idx]
                        title_idx += 1
                    else:
                        raise IndexError("Title vector index out of range.")
                if article["abstract"]:
                    if abstract_idx < len(abstract_vectors):
                        article["abstract_vector"] = abstract_vectors[abstract_idx]
                        abstract_idx += 1
                    else:
                        raise IndexError("Abstract vector index out of range.")

            # Prepare articles for bulk indexing and perform bulk indexing
            actions = [{"_index": "pubmed25_with_vector", "_id": article["pmid"], "_source": article} for article in articles]
            for chunk in chunker(actions, 50):
                bulk(es, list(chunk))

        # Move processed file to the processed directory
        processed_file_path = os.path.join(processed_directory, os.path.basename(file_path))
        move(file_path, processed_file_path)
        print(f"Moved processed file to {processed_file_path}")

    except (OSError, IOError) as e:
        print(f"Error reading file {file_path}: {e}")
        print(traceback.format_exc())  # Print the full stack trace
    except Exception as e:
        print(f"Error indexing file {file_path}: {e}")
        print(traceback.format_exc())  # Print the full stack trace