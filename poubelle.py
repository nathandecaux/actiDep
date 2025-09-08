import os
from actiDep.data.loader import Subject
from pprint import pprint
sub=Subject('01014')

pprint([f.path for f in sub.get(metric='FA',pipeline='anima_preproc')])