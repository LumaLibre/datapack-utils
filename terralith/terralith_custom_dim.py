#!/usr/bin/env python3
"""
terralith_custom_dim.py

Convert a Terralith datapack zip from its normal "globally override vanilla
overworld" form into a self-contained custom dimension under the namespace
`terralith_dim`. The output zip can be dropped into a Paper/Bukkit server's
world/datapacks/ folder, and the dimension can be loaded as a separate world
via Paper's bukkit.yml (dimension-type: terralith_dim:overworld).

USAGE
    python3 terralith_custom_dim.py <input.zip> [<output.zip>] [--folia]

If the output path is omitted, defaults to <input>_custom_dim.zip in the same
folder (or <input>_custom_dim_folia.zip if --folia is passed). The script is
idempotent: run it on a fresh Terralith zip from a new release and you get a
clean converted pack.

Pass --folia to also strip Terralith's .mcfunction files and function tags.
Folia rejects datapacks that contain functions due to its regionized
threading model. Terralith's functions are non-essential (intro message and
setup helpers) so dropping them has no impact on worldgen.

WHAT IT DOES
    1. Unpacks the input zip to a temp directory.
    2. Moves Terralith's data/minecraft/{worldgen,dimension,wolf_variant}
       overrides into data/terralith_dim/. (multi_noise_biome_source_parameter_list
       is intentionally NOT moved — it's a hardcoded Mojang enum.)
    3. Rewrites references throughout the pack with full registry-aware
       context. For every "minecraft:X" reference (and bare-name "X" which
       Minecraft implicitly treats as minecraft:X), checks if the target file
       has been moved into terralith_dim/ and rewrites to terralith_dim:X if
       so. Vanilla resources (blocks, items, mobs, untouched biomes) stay
       under minecraft:.
    4. Skips "type" fields (those are hardcoded enums, not registry refs).
    5. Forces dimension top-level "type" back to "minecraft:overworld" (the
       dimension_type registry, not noise_settings).
    6. Removes Terralith's two alternative full-world presets (all_skylands,
       amplified_large_biomes) that depend on the moved DF overrides and
       would otherwise fail registry validation. Cleans tag references too.
    7. Deletes data/minecraft/{worldgen,dimension,wolf_variant,tags} so the
       pack no longer modifies vanilla in ANY way. Tags in particular are
       removed because they bake Terralith memberships into chunk NBT;
       leaving them in causes chunk corruption when the datapack is later
       removed.
    8. Strips OS metadata files (.DS_Store, ._*) and zips it all back up.

AUTHOR NOTE
    Built iteratively against a real Paper 1.21.11 server (Canvas fork).
    Handles every Minecraft worldgen context I encountered in Terralith 2.6.0:
    - explicit minecraft: refs in dict values
    - bare-name implicit-minecraft refs (e.g., spline "coordinate" field)
    - lists (biome.carvers)
    - lists of lists (biome.features)
    - object-parent contexts (noise_settings.noise_router.*)
    - surface_rule biome_is predicate arrays
    If a future Terralith release introduces a new context where a ref hides,
    add the key to KEY_TO_REGISTRY (single-value), LIST_PARENT_TO_REGISTRY
    (list values), or OBJECT_PARENT_TO_REGISTRY (object whose every value
    is a ref regardless of inner key).

UNINSTALLING
    It is possible to uninstall this version of Terralith by removing the datapack and making some changes to worlds.
    - On 1.21.11: Remove the datapack, go into every world and modify the level.dat and remove the terralith_dim dimension.
    - On 26.1: Remove the datapack, delete the terralith_dim dimension folder.
    Biomes will be reset in the Terralith world and new chunks will not have the proper carvers or generation, but other worlds should be unaffected.
"""

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

NS = "terralith_dim"

# ---------------------------------------------------------------------------
# Registry-aware reference rewriting
# ---------------------------------------------------------------------------

# Worldgen registries that are extensible. We move files in these. Skipping
# multi_noise_biome_source_parameter_list because vanilla treats it as a
# hardcoded enum (only minecraft:overworld, :nether, :overworld_large_biomes
# are valid).
MOVE_WORLDGEN_CATS = [
    "biome",
    "configured_carver",
    "configured_feature",
    "density_function",
    "noise",
    "noise_settings",
    "placed_feature",
]

# JSON key whose direct value is a single reference to the named registry.
KEY_TO_REGISTRY = {
    "feature": "configured_feature",   # placed_feature.feature
    "argument": "density_function",
    "argument1": "density_function",
    "argument2": "density_function",
    "input": "density_function",
    "when_in_range": "density_function",
    "when_out_of_range": "density_function",
    "noise": "noise",                  # density_function.noise field
    "settings": "noise_settings",      # dimension.generator.settings
    "coordinate": "density_function",  # spline density function's coordinate
    "biome": "biome",                  # multi_noise parameter point .biome
}

# Parent key whose value is a list of refs.
LIST_PARENT_TO_REGISTRY = {
    "features": "placed_feature",      # biome.features = [[pf, ...], ...]
    "carvers": "configured_carver",    # biome.carvers = [cc, ...]
    "biomes": "biome",
    "biome_is": "biome",               # surface_rule.condition biome_is
}

# Object parent whose every value is a ref regardless of inner key name.
OBJECT_PARENT_TO_REGISTRY = {
    "noise_router": "density_function",
}

# Bare-name values we should NEVER rewrite even if they happen to match an ID.
# These are enum-like literals used in non-ref contexts.
ENUM_BARE_VALUES = {"none", "swamp", "dark_forest", "y", "x", "z"}


def determine_target_registry(parent_key, list_parent, object_parent):
    if parent_key in KEY_TO_REGISTRY:
        return KEY_TO_REGISTRY[parent_key]
    if list_parent in LIST_PARENT_TO_REGISTRY:
        return LIST_PARENT_TO_REGISTRY[list_parent]
    if object_parent in OBJECT_PARENT_TO_REGISTRY:
        return OBJECT_PARENT_TO_REGISTRY[object_parent]
    return None


def rewrite_refs(obj, inventory, parent_key=None, list_parent=None,
                 object_parent=None):
    """Recursively rewrite references in `obj` so that any minecraft:X (or
    bare-name X) pointing at a file we moved into terralith_dim/ becomes
    terralith_dim:X. Leaves everything else alone."""
    if isinstance(obj, dict):
        new_object_parent = parent_key
        for k, v in list(obj.items()):
            obj[k] = rewrite_refs(v, inventory, parent_key=k,
                                  list_parent=list_parent,
                                  object_parent=new_object_parent)
        return obj
    if isinstance(obj, list):
        # Preserve list_parent when descending into nested lists
        # (e.g. biome.features is list of list of strings).
        new_list_parent = parent_key if parent_key is not None else list_parent
        return [rewrite_refs(v, inventory, parent_key=None,
                             list_parent=new_list_parent,
                             object_parent=object_parent) for v in obj]
    if isinstance(obj, str):
        if parent_key == "type":
            return obj  # never rewrite hardcoded enum 'type' fields
        target = determine_target_registry(parent_key, list_parent,
                                           object_parent)
        if not target:
            return obj
        if obj.startswith("minecraft:"):
            rid = obj.split(":", 1)[1]
            if rid in inventory.get(target, set()):
                return f"{NS}:{rid}"
            return obj
        if ":" not in obj and obj not in ENUM_BARE_VALUES:
            if obj in inventory.get(target, set()):
                return f"{NS}:{obj}"
        return obj
    return obj


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def find_pack_root(extracted_dir):
    """Find the directory containing 'data/' inside the extracted zip.
    Some packs zip up the contents directly (data/ at root); others wrap
    everything in a top-level folder. Handle both."""
    if (extracted_dir / "data").is_dir():
        return extracted_dir
    # Look one level down
    for child in extracted_dir.iterdir():
        if child.is_dir() and (child / "data").is_dir():
            return child
    raise FileNotFoundError("Couldn't find 'data/' folder in the zip")


def convert(input_zip, output_zip, verbose=True, folia_compat=False):
    log = print if verbose else (lambda *a, **k: None)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extract_dir = tmpdir / "extracted"
        extract_dir.mkdir()

        # --- 1. Extract input zip ---
        log(f"Extracting {input_zip}...")
        with zipfile.ZipFile(input_zip, "r") as zf:
            zf.extractall(extract_dir)

        pack_root = find_pack_root(extract_dir)
        data_root = pack_root / "data"
        src = data_root / "minecraft"
        dst = data_root / NS

        # --- 2. Build inventory of files to move ---
        log("\nBuilding inventory of files to move...")
        inventory = {}
        for cat in MOVE_WORLDGEN_CATS:
            src_cat = src / "worldgen" / cat
            if not src_cat.is_dir():
                inventory[cat] = set()
                continue
            inventory[cat] = {
                f.relative_to(src_cat).with_suffix("").as_posix()
                for f in src_cat.rglob("*.json")
            }
            log(f"  {cat}: {len(inventory[cat])} files")

        # --- 3. Move files into terralith_dim/ ---
        log("\nMoving files into terralith_dim/...")
        moved_files = []
        for cat in MOVE_WORLDGEN_CATS:
            src_cat = src / "worldgen" / cat
            dst_cat = dst / "worldgen" / cat
            if not src_cat.is_dir():
                continue
            for src_file in src_cat.rglob("*.json"):
                rel = src_file.relative_to(src_cat)
                dst_file = dst_cat / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                moved_files.append(dst_file)

        # Dimension file
        src_dim = src / "dimension" / "overworld.json"
        dst_dim = dst / "dimension" / "overworld.json"
        if src_dim.exists():
            dst_dim.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dim, dst_dim)
            moved_files.append(dst_dim)

        # Wolf variant overrides
        src_wv = src / "wolf_variant"
        if src_wv.is_dir():
            dst_wv = dst / "wolf_variant"
            dst_wv.mkdir(parents=True, exist_ok=True)
            for src_file in src_wv.rglob("*.json"):
                rel = src_file.relative_to(src_wv)
                dst_file = dst_wv / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                moved_files.append(dst_file)

        log(f"  total files moved: {len(moved_files)}")

        # --- 4. Rewrite references inside moved files ---
        log("\nPass 1: rewriting references inside moved files...")
        pass1 = 0
        for f in moved_files:
            text = f.read_text(encoding="utf-8")
            if "minecraft:" not in text and ":" not in text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            before = json.dumps(obj, sort_keys=True)
            obj = rewrite_refs(obj, inventory)
            after = json.dumps(obj, sort_keys=True)
            if before != after:
                f.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                pass1 += (after.count(f'"{NS}:')
                          - before.count(f'"{NS}:'))
        log(f"  rewrote {pass1} references")

        # --- 5. Rewrite references in OTHER namespaces ---
        # Notably terralith/ files have references to vanilla
        # minecraft:overworld/* that we just moved.
        log("\nPass 2: rewriting references in other namespaces...")
        pass2 = 0
        pass2_files = 0
        for ns_dir in data_root.iterdir():
            if not ns_dir.is_dir() or ns_dir.name in ("minecraft", NS):
                continue
            for f in ns_dir.rglob("*.json"):
                text = f.read_text(encoding="utf-8")
                if "minecraft:" not in text and ":" not in text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    continue
                before = json.dumps(obj, sort_keys=True)
                obj = rewrite_refs(obj, inventory)
                after = json.dumps(obj, sort_keys=True)
                if before != after:
                    f.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                    pass2 += (after.count(f'"{NS}:')
                              - before.count(f'"{NS}:'))
                    pass2_files += 1
        log(f"  rewrote {pass2} references across {pass2_files} files")

        # --- 6. Force dimension_type back to vanilla ---
        if dst_dim.exists():
            obj = json.loads(dst_dim.read_text(encoding="utf-8"))
            if obj.get("type") != "minecraft:overworld":
                obj["type"] = "minecraft:overworld"
                dst_dim.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                log("\nForced dimension_type to minecraft:overworld")

        # --- 7. Delete overridden vanilla folders ---
        log("\nCleaning up data/minecraft/ overrides...")
        for sub in ["worldgen", "dimension", "wolf_variant"]:
            p = src / sub
            if p.exists():
                shutil.rmtree(p)
                log(f"  removed data/minecraft/{sub}/")

        # --- 8. Remove orphan Terralith world presets ---
        # all_skylands and amplified_large_biomes are alternative full-world
        # overhauls that depend on the moved density function overrides.
        # They're useless for a per-world custom dimension setup and would
        # fail registry validation. Their associated density_function
        # directory must also go.
        log("\nRemoving orphan Terralith full-world presets...")
        ORPHAN_PRESETS = ["all_skylands", "amplified_large_biomes"]
        terralith_dir = data_root / "terralith"
        for preset in ORPHAN_PRESETS:
            for sub in [terralith_dir / "worldgen" / "world_preset"
                        / f"{preset}.json",
                        terralith_dir / "worldgen" / "noise_settings"
                        / f"{preset}.json"]:
                if sub.exists():
                    sub.unlink()
                    log(f"  removed {sub.relative_to(data_root)}")
        orphan_df = (terralith_dir / "worldgen" / "density_function"
                     / "overworld_amplified_large_biomes")
        if orphan_df.exists():
            shutil.rmtree(orphan_df)
            log(f"  removed {orphan_df.relative_to(data_root)}/")

        # Clear orphan refs from world_preset tags
        for tagfile in data_root.rglob("tags/worldgen/world_preset/*.json"):
            try:
                obj = json.loads(tagfile.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            values = obj.get("values", [])
            new_values = [v for v in values if not any(
                f"terralith:{p}" == v for p in ORPHAN_PRESETS)]
            if new_values != values:
                obj["values"] = new_values
                tagfile.write_text(json.dumps(obj, indent=2),
                                   encoding="utf-8")
                log(f"  cleaned references in "
                    f"{tagfile.relative_to(data_root)}")

        # --- 8c. Isolate structures from vanilla biomes ---
        # Terralith structures are placed via structure_sets that target biome
        # tags like `terralith:has_structure/rubble_forest`. Those tags list
        # both Terralith biomes AND vanilla minecraft biomes (e.g. minecraft:
        # forest, minecraft:dark_forest). Since the structure placement runs
        # globally per-dimension based on biome matching, those structures
        # would still spawn in any world that contains the matching vanilla
        # biomes — including the user's main overworld. Once the datapack is
        # ever removed, those chunks have references to unknown structures
        # which spam warnings.
        #
        # Strip minecraft: biome refs from Terralith's has_structure tags.
        # Structures will still place in terralith_world (where the same
        # biomes now live under terralith_dim: namespace, plus all the
        # terralith: biomes), but no longer in vanilla worlds.
        log("\nIsolating structures from vanilla biomes...")
        structure_isolation_count = 0
        # Also rewrite to terralith_dim: variant of moved vanilla biomes so
        # the structure WILL place inside our custom dimension for those biomes
        moved_biomes = set()
        biome_dir = dst / "worldgen" / "biome"
        if biome_dir.is_dir():
            moved_biomes = {f.stem for f in biome_dir.rglob("*.json")}

        for tagfile in (data_root / "terralith" / "tags" / "worldgen"
                        / "biome" / "has_structure").rglob("*.json"):
            try:
                obj = json.loads(tagfile.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            values = obj.get("values", [])
            new_values = []
            changed = False
            for v in values:
                if isinstance(v, str) and v.startswith("minecraft:"):
                    rid = v.split(":", 1)[1]
                    if rid in moved_biomes:
                        # Vanilla biome we moved → reference the terralith_dim version
                        new_values.append(f"{NS}:{rid}")
                        changed = True
                        structure_isolation_count += 1
                    else:
                        # Vanilla biome we didn't move (exists only in vanilla worlds)
                        # → drop the reference entirely so structures don't place there
                        changed = True
                        structure_isolation_count += 1
                else:
                    new_values.append(v)
            if changed:
                obj["values"] = new_values
                tagfile.write_text(json.dumps(obj, indent=2),
                                   encoding="utf-8")
        log(f"  rewrote {structure_isolation_count} biome refs in "
            "has_structure tags")

        # --- 8d. Delete data/minecraft/tags/ entirely ---
        # Tags in the minecraft namespace are global — they apply to every
        # dimension and every world. They tell vanilla mechanics that
        # Terralith biomes/blocks belong in vanilla categories like
        # `#minecraft:is_forest`, `#minecraft:has_structure/village_plains`,
        # `#minecraft:mushroom_grow_block`, etc.
        #
        # The problem: once a chunk is saved with structures or biome
        # references that depend on these tag memberships, removing the
        # datapack leaves dangling references. Reports include corrupted
        # chunks after removal — the tag-augmented behavior gets baked
        # into NBT and breaks when the source tag goes away.
        #
        # The fix: delete the entire data/minecraft/tags/ tree. Trade-off:
        # - Inside terralith_world, vanilla structures (villages, ancient
        #   cities, mineshafts) won't generate in Terralith biomes
        # - Vanilla mob spawning rules keyed off biome tags won't apply
        #   to Terralith biomes
        # - Block-behavior tags lose Terralith's additions (e.g. mushrooms
        #   on extra block types)
        # In exchange: removing the datapack is now completely safe; no
        # chunk corruption, no dangling references in vanilla worlds.
        log("\nRemoving data/minecraft/tags/ to prevent chunk corruption "
            "on datapack removal...")
        mc_tags = src / "tags"
        if mc_tags.exists():
            n_files = sum(1 for _ in mc_tags.rglob("*.json"))
            shutil.rmtree(mc_tags)
            log(f"  removed data/minecraft/tags/ ({n_files} tag files)")

        # --- 8e. Optional Folia compatibility ---
        # Folia rejects datapacks containing .mcfunction files because of its
        # regionized threading model — functions assume a single global tick
        # scheduler that doesn't exist in Folia. Terralith ships only three
        # functions (a setup scoreboard init, an intro tellraw, and a debug
        # RTP helper), none of which affect worldgen. They can be deleted
        # without functional impact on the custom dimension.
        if folia_compat:
            log("\nFolia compatibility: removing function files...")
            n_removed = 0
            for func_dir in data_root.rglob("function"):
                if not func_dir.is_dir():
                    continue
                # Only the data/<namespace>/function dirs hold .mcfunction,
                # not data/<namespace>/tags/function (which contains tag JSON
                # files; those were already removed for the minecraft namespace
                # and are harmless to keep for terralith namespace, but Folia
                # ignores them).
                if func_dir.parent.name == "tags":
                    continue
                for f in func_dir.rglob("*.mcfunction"):
                    f.unlink()
                    n_removed += 1
                # Also remove function tag files in any non-minecraft namespace
                # so the function references don't dangle
            # Drop function tag files (which reference the now-deleted functions)
            for tag_func in data_root.rglob("tags/function/*.json"):
                tag_func.unlink()
                n_removed += 1
            log(f"  removed {n_removed} function/tag files")

        # --- 9. Strip OS metadata, empty dirs ---
        for trash_pattern in [".DS_Store", "._*"]:
            for p in pack_root.rglob(trash_pattern):
                p.unlink()
        # Repeat empty-dir cleanup until stable
        while True:
            empties = [d for d in pack_root.rglob("*")
                       if d.is_dir() and not any(d.iterdir())]
            if not empties:
                break
            for d in empties:
                d.rmdir()

        # --- 10. Validate ---
        log("\nValidating JSON...")
        errors = 0
        for f in data_root.rglob("*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    log(f"  parse error in {f}: {e}")
        if errors:
            log(f"  WARNING: {errors} JSON parse errors")
        else:
            log("  all JSON valid")

        # --- 11. Repackage ---
        log(f"\nWriting output zip to {output_zip}...")
        if Path(output_zip).exists():
            Path(output_zip).unlink()
        with zipfile.ZipFile(output_zip, "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            for f in pack_root.rglob("*"):
                if f.is_file():
                    name = f.name
                    if name == ".DS_Store" or name.startswith("._"):
                        continue
                    zf.write(f, f.relative_to(pack_root))

        # --- 12. Summary ---
        size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
        log(f"\nDone. Output: {output_zip} ({size_mb:.1f} MB)")
        log(f"  pass 1 (moved files): {pass1} refs rewritten")
        log(f"  pass 2 (other namespaces): {pass2} refs rewritten "
            f"across {pass2_files} files")
        log(f"  total: {pass1 + pass2} refs rewritten")


def main():
    args = sys.argv[1:]
    folia_compat = False
    if "--folia" in args:
        folia_compat = True
        args.remove("--folia")
    if len(args) < 1 or len(args) > 2:
        print(__doc__)
        sys.exit(1)
    input_zip = Path(args[0])
    if not input_zip.exists():
        sys.exit(f"Input zip not found: {input_zip}")
    if len(args) == 2:
        output_zip = Path(args[1])
    else:
        suffix = "_custom_dim_folia.zip" if folia_compat else "_custom_dim.zip"
        output_zip = input_zip.with_name(input_zip.stem + suffix)
    convert(input_zip, output_zip, folia_compat=folia_compat)


if __name__ == "__main__":
    main()