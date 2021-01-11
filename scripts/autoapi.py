import urllib3
import json
import time

from smp_audio.cmd import smp_audioArgumentParser
from smp_audio.util import ns2kw

api_url = 'http://127.0.0.1:5000'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {
    'User-Agent': user_agent,
    'X-Authentication': 'dt_78d052c4f2ac8cd26fe75709e108cb5a2bf3442f1da73310bcf746ab737b736f',
}

http = urllib3.PoolManager()

def main_autoedit_api(args):
    """main_autoedit_api

    Run autoedit workflow via API. Upload files, run the process,
    download output.
    """
    # pop overlisting
    args.filenames = args.filenames[0]
    
    # convert args to dict to json
    data = ns2kw(args)

    ############################################################
    # upload the files

    print(f"filenames {data['filenames']}")
    # would prefer this but heye
    # files = [('soundfile', (_, open(_, 'rb').read(), 'audio/wav')) for _ in data['filenames']]
    # print(f'files {files}')
    for filename in data['filenames']:
        files = [('soundfile', (filename, open(filename, 'rb').read(), 'audio/wav'))]
        # print(f'files {files}')
        
        call_url = api_url + '/files'
        r = http.request(
            'POST',
            call_url,
            # body=encoded_data,
            headers=headers,
            fields=files
        )
        res = json.loads(r.data.decode('utf-8'))
        # {'attribute': 'value'}
        # parse output
        # print(f'res {r.data}')
        print(f'res {res}')

    ############################################################
    # start the job
    print(f'data {data}')
    headers['Content-Type'] = 'application/json'
    # data = {'attribute': 'value'}
    call_url = api_url + '/api/smp/autoedit'
    print(f'call url {call_url}')
    print(f'call hdr {headers}')
    encoded_data = json.dumps(data).encode('utf-8')
    r = http.request(
        'POST',
        call_url,
        body=encoded_data,
        headers=headers
    )
    res = json.loads(r.data.decode('utf-8'))
    # {'attribute': 'value'}
    # parse output
    print(f'res {res}')

    location = res['data']['location']

    ############################################################
    # poll for output until 200 / not 404 or timeout
    time.sleep(30)

    # yeah
    del headers['Content-Type']
    
    # download output file
    call_url = api_url + '/files/' + location
    print(f'call url {call_url}')
    print(f'call hdr {headers}')
    r = http.request(
        'GET',
        call_url,
        headers=headers,
    )
    open(location, 'wb').write(r.data)
    print(f"downloaded {location}")
    return

def main(args):
    print(f'autoapi.main args {args}')
    if args.mode in ['autoedit']:
        res = main_autoedit_api(args)

    print(f'autoapi.main res {res}')

if __name__ == '__main__':
    parser = smp_audioArgumentParser()

    args = parser.parse_args()

    main(args)
