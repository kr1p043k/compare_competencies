import { motion } from "motion/react";
import { Award, BarChart3, Briefcase, GraduationCap } from "lucide-react";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

interface ProfileViewProps {
  data: {
    profile_name?: string;
    target_level?: string;
    skills_count?: number;
    skills?: string[];
    competencies_count?: number;
    competencies?: string[];
  };
}

const LEVEL_RU: Record<string, string> = {
  junior: "Junior",
  middle: "Middle",
  senior: "Senior",
};

export function ProfileView({ data }: ProfileViewProps) {
  const skills = data.skills || [];
  const show = skills.slice(0, 30);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Briefcase, label: "Профиль", value: data.profile_name || "–" },
          { icon: GraduationCap, label: "Уровень", value: LEVEL_RU[data.target_level || ""] || data.target_level || "–" },
          { icon: Award, label: "Навыков", value: String(data.skills_count ?? skills.length) },
          { icon: BarChart3, label: "Компетенций", value: String(data.competencies_count ?? (data.competencies || []).length) },
        ].map((s, i) => (
          <Card key={i} className="border border-slate-200 dark:border-slate-700">
            <CardContent className="pt-5">
              <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                <s.icon className="size-4" />
                <span className="text-xs font-medium">{s.label}</span>
              </div>
              <div className="mt-1 text-xl font-bold text-slate-900 dark:text-slate-100 truncate" title={s.value}>
                {s.value}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-base">
            <div className="p-2 bg-emerald-700 rounded-lg">
              <Award className="size-5 text-white" />
            </div>
            Навыки профиля
          </CardTitle>
          <CardDescription>
            {skills.length > show.length
              ? `Первые ${show.length} из ${skills.length} — полный список в gap-анализе`
              : `Всего: ${skills.length}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-1.5">
            {show.map((s) => (
              <Badge key={s} variant="outline" className="text-xs">
                {s}
              </Badge>
            ))}
            {show.length === 0 && (
              <p className="text-sm text-slate-500 dark:text-slate-400">Нет навыков.</p>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default ProfileView;
