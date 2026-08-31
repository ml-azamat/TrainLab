/**
 * shadcn/ui-style primitives: Radix behaviour + Tailwind styling, owned in-repo rather
 * than pulled from a component library, which is the pattern shadcn is built around.
 */
import * as React from "react";
import * as AccordionP from "@radix-ui/react-accordion";
import * as SelectP from "@radix-ui/react-select";
import * as SwitchP from "@radix-ui/react-switch";
import * as SliderP from "@radix-ui/react-slider";
import * as TabsP from "@radix-ui/react-tabs";
import * as TooltipP from "@radix-ui/react-tooltip";
import * as DialogP from "@radix-ui/react-dialog";
import * as PopoverP from "@radix-ui/react-popover";
import * as CheckboxP from "@radix-ui/react-checkbox";
import { Check, ChevronDown, X } from "lucide-react";
import { cn, parseNumber } from "@/lib/utils";

/* ------------------------------------------------------------------ Button */

const BTN = {
  default: "bg-[var(--color-accent)] text-white hover:brightness-110",
  outline: "border border-[var(--color-border)] bg-transparent hover:bg-[var(--color-panel2)]",
  ghost: "hover:bg-[var(--color-panel2)] text-[var(--color-muted)] hover:text-[var(--color-fg)]",
  danger: "bg-[var(--color-danger)] text-white hover:brightness-110",
  subtle: "bg-[var(--color-panel2)] border border-[var(--color-border)] hover:border-[var(--color-muted)]",
};
const SIZE = { sm: "h-6 px-2 text-[11px]", md: "h-8 px-3 text-xs", lg: "h-9 px-4 text-sm" };

export function Button({
  variant = "outline", size = "md", className, ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof BTN; size?: keyof typeof SIZE;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors",
        "disabled:opacity-40 disabled:pointer-events-none whitespace-nowrap",
        BTN[variant], SIZE[size], className,
      )}
      {...props}
    />
  );
}

/* ------------------------------------------------------------------ Input */

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-7 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)]",
        "px-2 text-xs tnum outline-none transition-colors",
        "focus:border-[var(--color-accent)] disabled:opacity-50",
        "placeholder:text-[var(--color-muted)]", className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

/* ------------------------------------------------------------------ NumberBox */

/**
 * Text box for a number.
 *
 * The typed text lives here rather than in the value it parses to. A controlled input fed
 * from the parsed number cannot represent a number that is still being typed: `Number("0.")`
 * is 0, so committing on every keystroke rewrote the box as "0" and swallowed the decimal
 * point — learning rate, weight decay and head init scale could only be given whole numbers.
 * Text that is not a number yet stays in the box and commits nothing, instead of being
 * coerced to 0 or stored as a string in a field the schema types as a float.
 *
 * `type="text"`, not `type="number"`: a number input renders 1e-5 as "0,00001" under
 * comma-decimal locales, which then parses back as NaN.
 */
export function NumberBox({
  value, onChange, integer, className, placeholder,
}: {
  value: unknown;
  onChange: (v: number | null) => void;
  integer?: boolean;
  className?: string;
  placeholder?: string;
}) {
  const [draft, setDraft] = React.useState<string | null>(null);
  return (
    <Input
      type="text" inputMode="decimal" className={className} placeholder={placeholder}
      value={draft ?? (value == null ? "" : String(value))}
      onChange={(e) => {
        setDraft(e.target.value);
        const parsed = parseNumber(e.target.value, integer);
        if (parsed !== undefined) onChange(parsed);
      }}
      // The draft has done its job once focus leaves: dropping it shows the value as it is
      // actually held ("0.0300" typed, 0.03 stored), and a half-typed number reverts.
      onBlur={() => setDraft(null)}
    />
  );
}

/* ------------------------------------------------------------------ Select */

export function Select({
  value, onValueChange, options, disabled, className, placeholder,
}: {
  value: string; onValueChange: (v: string) => void;
  options: { value: string; label?: string }[];
  disabled?: boolean; className?: string; placeholder?: string;
}) {
  return (
    <SelectP.Root value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectP.Trigger
        className={cn(
          "flex h-7 w-full items-center justify-between rounded-md border",
          "border-[var(--color-border)] bg-[var(--color-bg)] px-2 text-xs outline-none",
          "focus:border-[var(--color-accent)] disabled:opacity-50", className,
        )}
      >
        <SelectP.Value placeholder={placeholder} />
        <SelectP.Icon><ChevronDown className="h-3 w-3 opacity-50" /></SelectP.Icon>
      </SelectP.Trigger>
      <SelectP.Portal>
        <SelectP.Content
          position="popper" sideOffset={4}
          className="z-50 max-h-72 overflow-hidden rounded-md border border-[var(--color-border)] bg-[var(--color-panel2)] shadow-xl"
        >
          <SelectP.Viewport className="p-1">
            {options.map((o) => (
              <SelectP.Item
                key={o.value} value={o.value}
                className="relative flex cursor-pointer select-none items-center rounded px-6 py-1 text-xs outline-none data-[highlighted]:bg-[var(--color-accent)]/20"
              >
                <SelectP.ItemIndicator className="absolute left-1.5">
                  <Check className="h-3 w-3" />
                </SelectP.ItemIndicator>
                <SelectP.ItemText>{o.label ?? o.value}</SelectP.ItemText>
              </SelectP.Item>
            ))}
          </SelectP.Viewport>
        </SelectP.Content>
      </SelectP.Portal>
    </SelectP.Root>
  );
}

/* ------------------------------------------------------------------ Switch */

export function Switch({
  checked, onCheckedChange, disabled,
}: { checked: boolean; onCheckedChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <SwitchP.Root
      checked={checked} onCheckedChange={onCheckedChange} disabled={disabled}
      className={cn(
        "relative h-4 w-7 shrink-0 rounded-full border border-[var(--color-border)] transition-colors",
        "data-[state=checked]:bg-[var(--color-accent)] data-[state=unchecked]:bg-[var(--color-panel2)]",
        "disabled:opacity-40",
      )}
    >
      <SwitchP.Thumb className="block h-3 w-3 translate-x-0.5 rounded-full bg-white transition-transform data-[state=checked]:translate-x-3.5" />
    </SwitchP.Root>
  );
}

/* ------------------------------------------------------------------ Slider */

export function Slider({
  value, onValueChange, min = 0, max = 1, step = 0.01, disabled,
}: {
  value: number[]; onValueChange: (v: number[]) => void;
  min?: number; max?: number; step?: number; disabled?: boolean;
}) {
  return (
    <SliderP.Root
      value={value} onValueChange={onValueChange} min={min} max={max} step={step}
      disabled={disabled} className="relative flex h-4 w-full touch-none items-center"
    >
      <SliderP.Track className="relative h-[3px] w-full grow rounded-full bg-[var(--color-border)]">
        <SliderP.Range className="absolute h-full rounded-full bg-[var(--color-accent)]" />
      </SliderP.Track>
      {value.map((_, i) => (
        <SliderP.Thumb
          key={i}
          className="block h-3 w-3 rounded-full border-2 border-[var(--color-accent)] bg-[var(--color-bg)] outline-none hover:scale-110 transition-transform"
        />
      ))}
    </SliderP.Root>
  );
}

/* ------------------------------------------------------------------ Checkbox */

export function Checkbox({
  checked, onCheckedChange, disabled,
}: { checked: boolean; onCheckedChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <CheckboxP.Root
      checked={checked} onCheckedChange={(v) => onCheckedChange(Boolean(v))} disabled={disabled}
      className="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border border-[var(--color-border)] bg-[var(--color-bg)] data-[state=checked]:bg-[var(--color-accent)] data-[state=checked]:border-[var(--color-accent)] disabled:opacity-40"
    >
      <CheckboxP.Indicator><Check className="h-2.5 w-2.5 text-white" /></CheckboxP.Indicator>
    </CheckboxP.Root>
  );
}

/* ------------------------------------------------------------------ Tooltip */

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <TooltipP.Provider delayDuration={250} skipDelayDuration={100}>{children}</TooltipP.Provider>;
}

export function Tooltip({
  content, children, side = "left",
}: { content: React.ReactNode; children: React.ReactNode; side?: "top" | "right" | "bottom" | "left" }) {
  if (!content) return <>{children}</>;
  return (
    <TooltipP.Root>
      <TooltipP.Trigger asChild>{children}</TooltipP.Trigger>
      <TooltipP.Portal>
        <TooltipP.Content
          side={side} sideOffset={6} collisionPadding={12}
          className="z-50 max-w-xs rounded-md border border-[var(--color-border)] bg-[var(--color-panel2)] px-2.5 py-2 text-[11px] leading-relaxed text-[var(--color-fg)] shadow-xl"
        >
          {content}
          <TooltipP.Arrow className="fill-[var(--color-panel2)]" />
        </TooltipP.Content>
      </TooltipP.Portal>
    </TooltipP.Root>
  );
}

/* ------------------------------------------------------------------ Accordion */

export const Accordion = AccordionP.Root;
export const AccordionItem = AccordionP.Item;

export function AccordionTrigger({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <AccordionP.Header className="flex">
      <AccordionP.Trigger
        className={cn(
          "group flex flex-1 items-center gap-2 px-3 py-2 text-left outline-none transition-colors",
          "hover:bg-[var(--color-panel2)]", className,
        )}
      >
        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[var(--color-muted)] transition-transform duration-200 group-data-[state=open]:rotate-180" />
        {children}
      </AccordionP.Trigger>
    </AccordionP.Header>
  );
}

export function AccordionContent({ children }: { children: React.ReactNode }) {
  return (
    <AccordionP.Content className="overflow-hidden data-[state=open]:acc-open data-[state=closed]:acc-closed">
      <div className="px-3 pb-3 pt-1">{children}</div>
    </AccordionP.Content>
  );
}

/* ------------------------------------------------------------------ Tabs */

export const Tabs = TabsP.Root;
export const TabsContent = TabsP.Content;

export function TabsList({ children }: { children: React.ReactNode }) {
  return (
    <TabsP.List className="flex items-center gap-1 rounded-lg bg-[var(--color-panel)] p-0.5">
      {children}
    </TabsP.List>
  );
}

export function TabsTrigger({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <TabsP.Trigger
      value={value}
      className="rounded-md px-3 py-1 text-xs font-medium text-[var(--color-muted)] transition-colors data-[state=active]:bg-[var(--color-panel2)] data-[state=active]:text-[var(--color-fg)] hover:text-[var(--color-fg)]"
    >
      {children}
    </TabsP.Trigger>
  );
}

/* ------------------------------------------------------------------ Dialog / Popover */

export function Dialog({
  open, onOpenChange, title, children, wide,
}: {
  open: boolean; onOpenChange: (v: boolean) => void;
  title: string; children: React.ReactNode; wide?: boolean;
}) {
  return (
    <DialogP.Root open={open} onOpenChange={onOpenChange}>
      <DialogP.Portal>
        <DialogP.Overlay className="fixed inset-0 z-40 bg-black/60" />
        <DialogP.Content
          className={cn(
            "fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-full -translate-x-1/2 -translate-y-1/2",
            "overflow-auto rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4 shadow-2xl",
            wide ? "max-w-4xl" : "max-w-lg",
          )}
        >
          <div className="mb-3 flex items-center justify-between">
            <DialogP.Title className="text-sm font-semibold">{title}</DialogP.Title>
            <DialogP.Close asChild>
              <Button variant="ghost" size="sm"><X className="h-3.5 w-3.5" /></Button>
            </DialogP.Close>
          </div>
          {children}
        </DialogP.Content>
      </DialogP.Portal>
    </DialogP.Root>
  );
}

export const Popover = PopoverP.Root;
export const PopoverTrigger = PopoverP.Trigger;

export function PopoverContent({ children, className, align = "start" }: {
  children: React.ReactNode; className?: string; align?: "start" | "center" | "end";
}) {
  return (
    <PopoverP.Portal>
      <PopoverP.Content
        align={align} sideOffset={4}
        className={cn(
          "z-50 rounded-md border border-[var(--color-border)] bg-[var(--color-panel2)] p-2 shadow-xl",
          className,
        )}
      >
        {children}
      </PopoverP.Content>
    </PopoverP.Portal>
  );
}

/* ------------------------------------------------------------------ Badge */

const BADGE = {
  default: "bg-[var(--color-panel2)] text-[var(--color-muted)] border-[var(--color-border)]",
  accent: "bg-[var(--color-accent)]/15 text-[var(--color-accent)] border-[var(--color-accent)]/30",
  warn: "bg-[var(--color-modified)]/15 text-[var(--color-modified)] border-[var(--color-modified)]/30",
  danger: "bg-[var(--color-danger)]/15 text-[var(--color-danger)] border-[var(--color-danger)]/30",
  ok: "bg-[var(--color-ok)]/15 text-[var(--color-ok)] border-[var(--color-ok)]/30",
};

export function Badge({
  children, variant = "default", className,
}: { children: React.ReactNode; variant?: keyof typeof BADGE; className?: string }) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1 rounded border px-1.5 py-px text-[10px] font-medium",
      BADGE[variant], className,
    )}>
      {children}
    </span>
  );
}

export function Panel({
  title, children, actions, className,
}: { title?: React.ReactNode; children: React.ReactNode; actions?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-lg border border-[var(--color-border)] bg-[var(--color-panel)]", className)}>
      {title && (
        <div className="flex items-center justify-between border-b border-[var(--color-border)] px-3 py-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--color-muted)]">{title}</div>
          {actions}
        </div>
      )}
      <div className="p-3">{children}</div>
    </div>
  );
}
