import * as React from "react";
import { ListTree, Pencil } from "lucide-react";
import { api, type Json } from "@/lib/api";
import { Input, Select, Tooltip } from "./ui/primitives";

/**
 * Weight-set picker for the selected backbone.
 *
 * timm ships several weight sets per architecture (`convnext_tiny.in12k_ft_in1k`,
 * `.fb_in22k_ft_in1k`, …) and which ones exist is a property of the model, so the options
 * cannot live in the schema — they are fetched per backbone from `GET /api/backbones/{name}`
 * (`catalog.pretrained_tags`, which always leads with `default`).
 *
 * Anything the catalogue does not offer stays typeable, which is why this is a select with
 * a free-text escape rather than a plain enum: the list is only as good as the installed
 * timm, and a tag can arrive from a cloned run or a hand-written YAML. Refusing to show it
 * would quietly change which weights the run loads.
 */

export const DEFAULT_TAG = "default";

/**
 * The options to offer, given the fetched list and the value the config holds.
 *
 * The current value is always among them: a Radix Select whose value matches no item
 * renders its trigger empty, so a tag this build has not heard of would read as "no tag
 * set" while still being what the run trains with. Empty means "nothing to pick from" —
 * the caller falls back to free text.
 */
export function tagOptions(tags: string[] | null, current: string): string[] {
  if (!tags?.length) return [];
  return tags.includes(current) ? tags : [...tags, current];
}

export function PretrainedTagSelect({
  value, onChange, config,
}: { value: string; onChange: (v: string) => void; config: Json }) {
  const backbone: string = config?.model?.backbone ?? "";
  const [tags, setTags] = React.useState<string[] | null>(null);
  const [typing, setTyping] = React.useState(false);
  // What is in the text box, which is not always what is in the config: emptying the box
  // is a step on the way to typing a tag, but an empty tag would reach timm as a weight
  // set that does not exist, so the config keeps `default` until there is something real.
  const [draft, setDraft] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    setTags(null);
    if (!backbone) return;
    api.backbone(backbone)
      .then((d) => !cancelled && setTags(Array.isArray(d.tags) ? d.tags : null))
      .catch(() => !cancelled && setTags(null));
    return () => { cancelled = true; };
  }, [backbone]);

  const current = value || DEFAULT_TAG;
  const options = tagOptions(tags, current);
  const unlisted = options.length > 0 && !tags!.includes(current);

  // While the list is loading, and for a backbone whose catalogue entry did not load at
  // all, the field stays what it was before this control existed: a text box.
  if (typing || options.length === 0) {
    return (
      <div className="flex w-full min-w-0 items-center gap-1.5">
        <Input
          autoFocus={typing}
          value={draft ?? value ?? ""} placeholder={DEFAULT_TAG}
          onChange={(e) => {
            setDraft(e.target.value);
            onChange(e.target.value === "" ? DEFAULT_TAG : e.target.value);
          }}
        />
        {options.length > 0 && (
          <Tooltip content={`Pick from the ${tags!.length} weight sets timm has for ${backbone}`}>
            <button
              onClick={() => { setDraft(null); setTyping(false); }}
              className="shrink-0 text-[var(--color-muted)] hover:text-[var(--color-fg)]"
            >
              <ListTree className="h-3 w-3" />
            </button>
          </Tooltip>
        )}
      </div>
    );
  }

  return (
    <div className="flex w-full min-w-0 flex-col gap-0.5">
      <div className="flex min-w-0 items-center gap-1.5">
        <Select
          value={current} onValueChange={onChange}
          options={options.map((t) => ({ value: t }))}
        />
        <Tooltip content="Type a tag by hand">
          <button
            onClick={() => { setDraft(value ?? ""); setTyping(true); }}
            className="shrink-0 text-[var(--color-muted)] hover:text-[var(--color-fg)]"
          >
            <Pencil className="h-3 w-3" />
          </button>
        </Tooltip>
      </div>
      {unlisted && (
        <span className="pl-0.5 text-[10px] text-[var(--color-modified)]">
          not among {backbone}'s weight sets
        </span>
      )}
    </div>
  );
}
