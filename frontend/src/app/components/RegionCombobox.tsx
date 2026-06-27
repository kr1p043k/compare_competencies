import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "../components/ui/utils";
import { Button } from "../components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "../components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../components/ui/popover";

interface Region {
  id: number;
  name: string;
}

interface RegionComboboxProps {
  regions: Region[];
  value: string;
  onChange: (value: string) => void;
}

export function RegionCombobox({
  regions,
  value,
  onChange,
}: RegionComboboxProps) {
  const [open, setOpen] = useState(false);

  const selectedRegion = regions.find((r) => r.id.toString() === value);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between h-9"
        >
          {selectedRegion ? selectedRegion.name : "Выберите регион..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0" align="start">
        <Command>
          <CommandInput placeholder="Поиск региона..." />
          <CommandList>
            <CommandEmpty>Регион не найден.</CommandEmpty>
            <CommandGroup>
              {regions.map((region) => (
                <CommandItem
                  key={region.id}
                  value={region.name}
                  keywords={[region.id.toString()]}
                  onSelect={() => {
                    onChange(region.id.toString());
                    setOpen(false);
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      value === region.id.toString()
                        ? "opacity-100"
                        : "opacity-0"
                    )}
                  />
                  {region.name}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
