from pprint import pformat
def slice2track(slicetable, tracktable):
    """slice to track data mapper

    files_tracks = [_.rstrip() for _ in open('/home/lib/audio/work/fm_2019_sendspaace/data/files_20190209_unique
        : .txt', 'r').readlines()]

    files_tracks_ = [_.split('/')[-1][:-4] for _ in files_tracks]

    tracktable = pd.DataFrame(files_tracks_, columns=['filename'])


    tracktable_ = pd.read_excel('/home/lib/audio/work/fm_2019_sendspaace/data/files_20190209_unique.xlsx')

    """
    sliceindex = list(slicetable.index)
    # print(pformat(sliceindex))
    # trackrefs = [_[:-4] for _ in tracktable.File]
    trackrefs = [_ for _ in tracktable.File]
    print('trackrefs: {0}'.format(pformat(trackrefs)))

    tracklist = []
    for index, row in slicetable.iterrows():
        print('index: {0}'.format(index))
        print('  row: {0}'.format(row['index']))
    
        # trackmatch = [_ for _ in trackrefs if _[:-4] in row['index']]
        trackmatch = [i for i,_ in enumerate(trackrefs) if _[:-4] in row['index']]
        # trackmatch = [_[:-4] for _ in tracktable_.File]
        if len(trackmatch) > 0:
            # print('    trackmatch: {0}'.format(trackmatch))
            # print('position: {0}, {1}'.format(index, tracktable.iloc[trackmatch[0]][['Artist', 'File']]))
            tracklist_item = dict(tracktable.iloc[trackmatch[0]][['Artist', 'Track']])
            tracklist_item['pos'] = '{0: .2f}'.format(row['numseconds_cum'] - row['numseconds'])
            tracklist_item['length'] = '{0: .2f}'.format(row['numseconds'])
            if index > 0:
                tracklist_item['seg'] = '{0}'.format(row['index'][-13:][:-4])
            else:
                tracklist_item['seg'] = '0000.000000'
            print('{0}'.format(tracklist_item))
            tracklist.append(tracklist_item)
    return tracklist
