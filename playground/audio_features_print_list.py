import re

# audio files
filearray = [
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Powell--New-Beta-Vol.1--Boomkat(4).mp3', 'mp3', 5292000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Autechre - spl47.mp3', 'mp3', 14553000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//03_db_247927330_soundcloud.mp3', 'mp3', 17992800, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//10. Joy Division - I Remember Nothing (1979).mp3', 'mp3', 15567300, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//05. Original Degung Instrumentalia - Sudanese Gamelan Music - [Panineungan].mp3', 'mp3', 17551800, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//1going_320.mp3', 'mp3', 6174000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Burial - Beachfires.mp3', 'mp3', 26063100, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//02-magic.ii.mp3', 'mp3', 23152500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//02 Man Machine (Kraftwerk).mp3', 'mp3', 4586400, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//The Beach Boys - Heroes And Villains Sections  (Bonus Track. Stere.mp3', 'mp3', 19227600, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//11-prefuse_73-we_got_our_own_way_feat._kazu-ftd.mp3', 'mp3', 8775900, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Stokes Croft-001-Kamikaze Space Programme-Choke (Original Mix).mp3', 'mp3', 15743700, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//05 FriscoBum.mp3', 'mp3', 9922500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//07-Bricc_Baby_Shitro-IDK_Feat_Casey_Veggies_Prod_By_Metro_Boomin.mp3', 'mp3', 13009500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//a deepness upon the sky ~by micrOmega [soundtake.net].mp3', 'mp3', 18963000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//militik ~by micrOmega [soundtake.net].mp3', 'mp3', 9702000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//The Beach Boys - Heroes And Villains.mp3', 'mp3', 12877200, 44100),
]

# audio files
filearray2 = [
    ('Powell - Freezer (New Beta Vol. 1)', 'mp3', 5292000, 44100),
    ('Autechre - spl47', 'mp3', 14553000, 44100),
    ('/DL/MS/ - yep (03_db_247927330_soundcloud)', 'mp3', 17992800, 44100),
    ('Joy Division - I Remember Nothing (1979)', 'mp3', 15567300, 44100),
    ('Original Degung Instrumentalia - Panineungan (Sudanese Gamelan Music)', 'mp3', 17551800, 44100),
    ('tsx - going (recur)', 'mp3', 6174000, 44100),
    ('Burial - Beachfires', 'mp3', 26063100, 44100),
    ('fushitsusha - 02 magic ii', 'mp3', 23152500, 44100),
    ('Yat Kha - 02 Man Machine (Kraftwerk)', 'mp3', 4586400, 44100),
    ('The Beach Boys - Heroes And Villains Sections  (Bonus Track. Stere', 'mp3', 19227600, 44100),
    ('Prefuse73 - We got our own way feat. Kazu', 'mp3', 8775900, 44100),
    ('Kamikaze Space Programme - Choke (Original Mix) (Stokes Croft)', 'mp3', 15743700, 44100),
    ('Limonious - FriscoBum', 'mp3', 9922500, 44100),
    ('Bricc Baby Shitro - IDK Feat. Casey Veggies, Prod. By Metro Boomin', 'mp3', 13009500, 44100),
    ('micrOmega - a deepness upon the sky', 'mp3', 18963000, 44100),
    ('micrOmega - militik', 'mp3', 9702000, 44100),
    ('The Beach Boys - Heroes And Villains', 'mp3', 12877200, 44100),
]

for fl in filearray2:
    # print re.sub(r'\/home\/lib\/audio\/work\/fm_mix_subcity_sorbierd_20171125\/\/', r'', fl[0])
    # flr = re.sub(r'\/home\/lib\/audio\/work\/fm_mix_subcity_sorbierd_20171125\/\/', r'', fl[0])
    print(fl[0])
