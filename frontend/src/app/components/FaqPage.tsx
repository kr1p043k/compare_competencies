import { useState, useMemo } from "react";
import { Input } from "./ui/input";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "./ui/accordion";
import { HelpCircle, Search } from "lucide-react";
import { Badge, Card, CardBody, EmptyState } from "../../design";

import { faqData, type FaqItem } from "./FaqPage.data";

export function FaqPage() {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    if (!search.trim()) return faqData;
    const q = search.toLowerCase();
    return faqData.filter(
      (f) => f.q.toLowerCase().includes(q) || f.a.toLowerCase().includes(q) || f.category.toLowerCase().includes(q),
    );
  }, [search]);

  const grouped = useMemo(() => {
    const map = new Map<string, FaqItem[]>();
    for (const item of filtered) {
      if (!map.has(item.category)) map.set(item.category, []);
      map.get(item.category)!.push(item);
    }
    return map;
  }, [filtered]);

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center gap-3 border-b border-border bg-muted px-6 py-5">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <HelpCircle className="size-5" aria-hidden="true" />
        </div>
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-foreground">
            Помощь и FAQ
          </h2>
          <p className="text-sm text-muted-foreground">
            Ответы на частые вопросы о платформе
          </p>
        </div>
      </div>
      <CardBody className="space-y-6 p-6">
        <div className="relative">
          <Search
            aria-hidden="true"
            className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <label htmlFor="faq-search" className="sr-only">
            Поиск по вопросам
          </label>
          <Input
            id="faq-search"
            placeholder="Поиск по вопросам..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-11 pl-10"
          />
        </div>

        <div className="space-y-6">
          {Array.from(grouped.entries()).map(([category, items]) => (
            <div key={category}>
              <div className="flex items-center gap-2 mb-3">
                <Badge tone="neutral" className="px-3 py-1 text-xs font-medium">
                  {category}
                </Badge>
                <span className="text-xs text-muted-foreground">{items.length}</span>
              </div>
              <Accordion type="single" collapsible className="divide-y divide-border overflow-hidden rounded-lg border border-border">
                {items.map((item, idx) => (
                  <AccordionItem key={idx} value={`${category}-${idx}`} className="border-none">
                    <AccordionTrigger className="px-4 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted hover:no-underline">
                      {item.q}
                    </AccordionTrigger>
                    <AccordionContent className="px-4 text-sm leading-relaxed text-muted-foreground">
                      {item.a}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          ))}

          {filtered.length === 0 && (
            <EmptyState
              icon={<HelpCircle className="size-12 opacity-40" aria-hidden="true" />}
              title={<>Ничего не найдено по запросу «{search}»</>}
            />
          )}
        </div>
      </CardBody>
    </Card>
  );
}
