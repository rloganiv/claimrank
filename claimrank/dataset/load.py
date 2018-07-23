"""Load serialized data into Python."""

import csv
import json
from collections import namedtuple


# TODO: Filter out trailing tabs from .pm file.
PMInstance = namedtuple('PMInstance', ['input_sentence', 'entity_name',
                                       'post_modifier', 'gold_sentence',
                                       'wiki_id', 'previous_sentence',
                                       'blank'])
WikiInstance = namedtuple('WikiInstance', ['wiki_id', 'entity_name', 'aliases',
                                           'description', 'claims'])
Claim = namedtuple('Claim', ['field_name', 'value', 'qualifiers'])


def load_pm(filename):
    """Reads a .pm file into a list of PMInstances.

    Parameters
    ----------
    filename : str
        Path to the .pm file.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        out = [PMInstance(*row) for row in reader]
    return out


def load_wiki(filename):
    """Reads a .wiki file into a dictionary whose keys are ``wiki_id``s and
    whose value are ``WikiInstance``s.

    Parameters
    ----------
    filename : str
        Path to the .wiki file.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        out = dict()
        for line in reader:
            wiki_id = line[0]
            entity_name = line[1]
            aliases = line[2]
            description = line[3]
            unprocessed_claims = json.loads(line[4])
            processed_claims = []
            for unprocessed_claim in unprocessed_claims:
                field_name, value = unprocessed_claim['property']
                qualifiers = unprocessed_claim['qualifiers']
                processed_claim = Claim(field_name, value, qualifiers)
                processed_claims.append(processed_claim)
            out[wiki_id] = WikiInstance(wiki_id, entity_name, aliases,
                                        description, processed_claims)
    return out

