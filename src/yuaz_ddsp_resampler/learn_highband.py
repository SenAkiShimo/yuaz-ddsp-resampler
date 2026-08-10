#!/usr/bin/env python3

import argparse,json,shutil,time

from pathlib import Path

from .learned_highband import build_profile_database,save_profile_database


def main():

    p=argparse.ArgumentParser()

    p.add_argument('voicebank')

    p.add_argument('--project-root',required=True)

    p.add_argument('--force',action='store_true',help='discard alpha.8 high-band cache and re-analyze source WAVs')

    a=p.parse_args()

    project=Path(a.project_root).resolve()

    bank=Path(a.voicebank).expanduser().resolve()

    state=bank/'.yuaz-alpha8-rc3-2'

    manifest_path=state/'manifest.json'

    if not manifest_path.exists():

        raise RuntimeError('No .yuaz-alpha8-rc3-2/manifest.json found. Run alpha.8 Prepare Voicebank first.')

    if a.force:

        cache=state/'highband_cache_v3'

        if cache.exists():

            shutil.rmtree(cache)

        print('Cleared alpha.8 High-Band cache; source WAVs will be re-analyzed.')

    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))

    db=build_profile_database(bank,manifest.get('entries') or [],model_hop=256,model_sr=24000,state_dir=state)

    out=state/'highband_profiles_v3.json'

    save_profile_database(out,db)

    bank_id=(manifest.get('profile') or {}).get('voicebank_id')

    registry_path=project/'voicebank_registry.json'

    if not registry_path.exists():

        raise RuntimeError('No alpha.8 registry found. Run Prepare Voicebank first.')

    registry=json.loads(registry_path.read_text(encoding='utf-8'))

    changed=0

    for record in (registry.get('samples') or {}).values():

        if isinstance(record,dict) and (not bank_id or record.get('voicebank_id')==bank_id):

            r=record.get('voicebank_root')

            if r and Path(r).expanduser().resolve()==bank:

                record['highband_profiles']=str(out.resolve())

                record['highband_profile_format']=int(db.get('format',3))

                changed+=1

    registry['updated_at_highband_v3']=time.time()

    registry_path.write_text(json.dumps(registry,indent=2,ensure_ascii=False),encoding='utf-8')

    stats=db.get('stats') or {}

    print(f"Learned High Band v3 continuity: {stats.get('alias_count',0)} aliases; {stats.get('analyzed',0)} analyzed, {stats.get('cached',0)} cached, {stats.get('skipped',0)} skipped.")

    print(f"Temporal prototypes: {stats.get('temporal_prototypes',0)}")

    print(f"Saved: {out}")

    print(f"Updated {changed} alpha.8 registry records.")

    print('Other adaptation states and source WAVs were not modified.')


if __name__=='__main__':

    main()

