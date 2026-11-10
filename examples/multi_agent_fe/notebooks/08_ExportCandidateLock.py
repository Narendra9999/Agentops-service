# Databricks notebook source
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
queue=[Requirement(line) for line in (root/'requirements.in').read_text().splitlines() if line and not line.startswith('#')]
selected={}
while queue:
    requirement=queue.pop()
    name=canonicalize_name(requirement.name)
    extras=set(requirement.extras)
    previous=selected.get(name)
    if previous is not None and extras.issubset(previous):
        continue
    selected[name]=(previous or set()) | extras
    dist=metadata.distribution(name)
    if not requirement.specifier.contains(dist.version,prereleases=True):
        raise RuntimeError(f'Installed dependency {name} {dist.version} violates {requirement}')
    for raw in dist.requires or []:
        dependency=Requirement(raw)
        if dependency.marker is None or any(dependency.marker.evaluate({'extra':extra}) for extra in (selected[name] | {''})):
            queue.append(dependency)
lock='\n'.join(f'{name}=={metadata.version(name)}' for name in sorted(selected))+'\n'
(root/'requirements.lock').write_text(lock)

dbutils.notebook.exit(json.dumps({'lock':lock,'python':sys.version}))
