"""Isolated G4 transport: fixed input vertex, W density and tau-helicity proxies."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from analysis.flavour.inventory import write_json


LCG107 = '/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt'
G4_SOURCE = Path('/eos/home-x/xcheng/FASER/Pinpoint/Pinpoint_G4/Pinpoint')


def replace_once(path, original, changed):
    text = path.read_text()
    if text.count(original) != 1:
        raise ValueError(f'G4 source patch does not match {path.name}')
    path.write_text(text.replace(original, changed))


def lcg107(command, **kwargs):
    script = '\n'.join(['which() { command -v "$@"; }', f'source {LCG107}/setup.sh', 'set -e',
                         'export QT_QPA_PLATFORM=offscreen', f'export PYTHIA8DATA={LCG107}/share/Pythia8/xmldoc',
                         'exec '+shlex.join(list(map(str, command)))])
    return subprocess.run(['/bin/bash', '-c', script], check=True, **kwargs)


def build(args):
    started = time.monotonic()
    destination = args.root/'g4'
    destination.mkdir(exist_ok=True)
    scratch = Path(os.environ.get('PINPOINT_S6_SCRATCH', '/tmp'))/'g4_build'
    source = scratch/'source'
    source.mkdir(parents=True, exist_ok=True)
    for name in ['include', 'src', 'macros']:
        shutil.copytree(args.g4_source/name, source/name, dirs_exist_ok=True)
    for name in ['Pinpoint.cc', 'CMakeLists.txt']:
        shutil.copyfile(args.g4_source/name, source/name)
    generator = source/'src/generators/GFaserGenerator.cc'
    replace_once(generator, '#include <string>', '#include <string>\n#include <cstdlib>')
    replace_once(generator, '  fVz = GenerateRandomZVertex();',
        '  const char* s6InputZ = std::getenv("PINPOINT_USE_INPUT_Z");\n'
        '  if (!s6InputZ || std::string(s6InputZ) != "1") fVz = GenerateRandomZVertex();')
    detector = source/'src/DetectorConstruction.cc'
    replace_once(detector, '#include <iomanip>', '#include <iomanip>\n#include <cstdlib>\n#include "G4Material.hh"')
    replace_once(detector, '  G4Material* tungstenMaterial = nist->FindOrBuildMaterial("G4_W");',
        '  G4Material* tungstenMaterial = nist->FindOrBuildMaterial("G4_W");\n'
        '  if (const char* s6ScaleText = std::getenv("PINPOINT_W_DENSITY_SCALE")) {\n'
        '    const double s6Scale = std::atof(s6ScaleText);\n'
        '    tungstenMaterial = new G4Material("S6_W_density", tungstenMaterial->GetDensity()*s6Scale, tungstenMaterial);\n'
        '    G4cout << "S6 tungsten density scale = " << s6Scale << G4endl;\n'
        '  }')
    decayer = source/'src/Py8Decayer.cc'
    replace_once(decayer, '#include "Py8Decayer.hh"', '#include "Py8Decayer.hh"\n#include <cstdlib>')
    replace_once(decayer,
        '  fDecayer->event.back().pol(\n    round(std::cos(track.GetPolarization().angle(track.GetMomentumDirection()))));',
        '  double s6Polarization = round(std::cos(track.GetPolarization().angle(track.GetMomentumDirection())));\n'
        '  if (std::abs(pdgid) == 15) {\n'
        '    if (const char* s6Helicity = std::getenv("PINPOINT_TAU_HELICITY"))\n'
        '      s6Polarization = std::atof(s6Helicity)*(pdgid > 0 ? 1. : -1.);\n'
        '  }\n'
        '  fDecayer->event.back().pol(s6Polarization);')
    xml = Path(LCG107)/'share/Pythia8/xmldoc'
    replace_once(decayer, 'new Pythia("../share/Pythia8/xmldoc", false)', f'new Pythia("{xml}", false)')
    archive = destination/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(source, arcname='source')
    with (destination/'build.log').open('w') as log:
        lcg107(['cmake', '-S', source, '-B', scratch/'build'], stdout=log, stderr=subprocess.STDOUT)
        lcg107(['cmake', '--build', scratch/'build', '--parallel', args.cpus], stdout=log, stderr=subprocess.STDOUT)
    shutil.copyfile(scratch/'build/pinpoint', destination/'pinpoint_s6')
    (destination/'pinpoint_s6').chmod(0o755)
    shutil.copytree(source/'macros', destination/'macros', dirs_exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(args.g4_source), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(destination/'build.json', {'status': 'complete', 'source': str(args.g4_source), 'source_commit': commit,
        'LCG': LCG107, 'seconds': time.monotonic()-started, 'cpus': args.cpus,
        'patches': ['optional fixed input z; original default preserved', 'optional W material density; geometry unchanged',
                    'optional charge-conjugate tau helicity, no charm override', 'absolute matching Pythia XML directory'],
        'tau_helicity_convention': 'parameter is tau-minus helicity; tau-plus gets opposite helicity',
        'original_source_modified': False, 'UTC': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'build': 'complete', 'seconds': time.monotonic()-started}), flush=True)


def macro_text(input_file, output_file, geometry_macro, events, field=.57, seed=20261114):
    return '\n'.join(['/control/verbose 0', '/run/verbose 0', f'/control/execute {geometry_macro}',
        '/det/enableFaserSpectrometer true', '/gen/select gfaser', f'/gen/gfaser/inputFile {input_file}',
        '/gen/gfaser/hitPixelArea false', f'/out/fileName {output_file}', '/out/saveTrack false',
        '/out/saveTruthHits false', f'/random/setSeeds {seed} {seed+17}', '/run/initialize',
        f'/Pinpoint/field/value {field} tesla', f'/run/beamOn {events}'])+'\n'


def smoke(args):
    import uproot
    import numpy as np
    scratch = Path(os.environ.get('PINPOINT_S6_SCRATCH', '/tmp'))/'g4_smoke'
    scratch.mkdir(parents=True, exist_ok=True)
    output = scratch/'smoke.root'
    macro = scratch/'smoke.mac'
    macro.write_text(macro_text('/eos/project-f/fasersim-bonn/public/pinpoint/2025-11-26-genie/nue.gfaser.root',
        output, args.root/'g4/macros/geom.mac', 1))
    env = dict(os.environ, PINPOINT_USE_INPUT_Z='0')
    with (args.root/'g4/smoke.log').open('w') as log:
        lcg107([args.root/'g4/pinpoint_s6', macro], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT)
    expected = json.loads(Path(json.loads((args.root/'plan.json').read_text())['geometry']).read_text())
    with uproot.open(output) as file:
        actual = file['geometry'].arrays(library='ak')[0].to_list()
        comparisons = {key: bool(np.allclose(actual[key], expected[key], rtol=2e-6, atol=2e-4))
            if np.asarray(actual[key]).shape == np.asarray(expected[key]).shape else False for key in expected}
        result = {'events': file['event'].num_entries, 'geometry_comparisons': comparisons,
            'status': 'passed' if all(comparisons.values()) else 'geometry_review_required', 'test_used': False,
            'definition': 'interface geometry compatibility for resimulation, not re-certification of accepted samples'}
    write_json(args.root/'g4/smoke.json', result)
    output.unlink()
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('resimulation geometry must match the supplied observation geometry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'smoke'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--g4-source', type=Path, default=G4_SOURCE)
    parser.add_argument('--cpus', type=int, default=4)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
