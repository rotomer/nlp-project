import requests
from requests import HTTPError


def index_relations_for_file(input_file_path, output_folder_path):
    response = requests.post(
        'http://localhost:9500/indexRelations',
        json={'inputFilePath': input_file_path,
              'outputFolderPath': output_folder_path})

    if response.status_code != 200:
        raise HTTPError(
            'Failed to call Stanford NLP Server to index relations. Status code' + str(response.status_code))
