export interface PublicationFileRef {
  url: string;
  filename: string | null;
  size: number | null;
  content_type: string | null;
}

/** Плоская схема публикации хаба (совпадает с PublicationOut на бэкенде). */
export interface Publication {
  id: number;
  title: string | null;
  authors: string[];
  type_id: number | null;
  type_name: string | null;
  signature: string | null;
  doi: string | null;
  isbn: string | null;
  track_id: string | null;
  created_at: string | null;
  year: number | null;
  academic_subject: string[];
  education_program: string[];
  abstract: string | null;
  keywords: string[];
  journal: string | null;
  publisher: string | null;
  language: string | null;
  publishing_year: number | null;
  pages: string | null;
  files: PublicationFileRef[];
  url: string | null;
  web_url: string | null;
  creator_username: string | null;
}

/**
 * Локальный фильтр по уже загруженным записям.
 *
 * У hub.sfedu.ru нет серверного поиска, поэтому фильтруем на клиенте.
 * Каждое слово запроса должно встретиться хотя бы в одном поле.
 */
export function matchesQuery(pub: Publication, needle: string): boolean {
  const normalized = needle.trim().toLowerCase();
  if (!normalized) return true;
  const haystack = [
    pub.title ?? "",
    pub.type_name ?? "",
    pub.authors.join(" "),
    pub.academic_subject.join(" "),
    pub.education_program.join(" "),
    pub.keywords.join(" "),
    pub.journal ?? "",
    pub.publisher ?? "",
    pub.abstract ?? "",
    pub.doi ?? "",
  ]
    .join(" ")
    .toLowerCase();
  return normalized.split(/\s+/).every((part) => haystack.includes(part));
}

/** Обрезает аннотацию для карточки, не разрывая слово. */
export function shortenAbstract(text: string | null, limit = 240): string | null {
  if (!text) return null;
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}...`;
}

/** Ссылка на карточку в хабе: web_url открывает UI, url ведёт на API и просит токен. */
export function resolveHubLink(pub: Publication): string | null {
  return pub.web_url ?? pub.url;
}
