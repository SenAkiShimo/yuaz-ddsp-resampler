#!/usr/bin/env python3
import argparse
from .transaction import run_transaction

def main():
    p=argparse.ArgumentParser()
    p.add_argument('voicebank')
    p.add_argument('--project-root', required=True)
    p.add_argument('--force', action='store_true', help='Compatibility option; RC3.3 high-band rebuild is always transactional.')
    a=p.parse_args()
    run_transaction(a.project_root, a.voicebank, 'highband')

if __name__=='__main__':
    main()
