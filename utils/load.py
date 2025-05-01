import json
def load_minions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    minion_list = []
    for entry in data.get("data", []):
        name = entry.get("name", "")
        types = entry.get("types", [])
        tier = entry.get("tier", -1)
        attack = entry.get("attack", 0)
        health = entry.get("health", 0)

        if not name or tier == -1:
            continue  # skip malformed entries

        minion_list.append({
            "name": name,
            "types": types,
            "attack": attack,
            "health": health, 
            "tier": tier
        })

    return minion_list

# Example usage
if __name__ == "__main__":
    path = "data\\bg_minions_all.json"  # your full file
    minions = load_minions(path)

    for m in minions:
        if m.get('name').startswith('Brann'):
            for tribe in m.types or []:
                if tribe in TRIBE_TO_INDEX:
                    tribes[TRIBE_TO_INDEX[tribe]] = 1
