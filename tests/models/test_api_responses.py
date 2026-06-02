from src.models.api_responses import (
    CacheRefreshResponse,
    CategoryCoverage,
    ClusterSummaryItem,
    ClusterSummaryResponse,
    ClustersByLevelResponse,
    DeadSkillsResponse,
    GapProgressResponse,
    HealthResponse,
    KRMCompetency,
    KRMCoverageResponse,
    KRMExpertiseItem,
    LevelClusters,
    MarketCompetenciesResponse,
    MissingSkillItem,
    MissingSkillsResponse,
    PipelineSimpleResponse,
    PipelineStatusResponse,
    PipelineTaskListResponse,
    PipelineTaskStatus,
    ProfessionDetailResponse,
    ProfessionEvalResponse,
    ProfessionItem,
    ProfessionsResponse,
    ProfileShort,
    ProfiledEval,
    ProfilesCompareResponse,
    ReadyResponse,
    RegionsResponse,
    SkillInfoResponse,
    SkillItem,
    StatusResponse,
    TaxonomyCoverageResponse,
    TopSkillsResponse,
    TrendsResponse,
    VacanciesByRegionResponse,
    VacanciesResponse,
    VacancyDetailResponse,
    VacancyItem,
    VacancyStatsResponse,
)


class TestApiResponses:
    def test_health_response(self):
        r = HealthResponse(status="ok", version="1.0", evaluator=True, recommendation_engine=True)
        assert r.status == "ok"

    def test_ready_response(self):
        r = ReadyResponse(status="ready", components={"db": True})
        assert r.status == "ready"

    def test_status_response(self):
        r = StatusResponse(
            vacancies_loaded=True, skill_weights_count=10, taxonomy_loaded=True,
            whitelist_size=100, profiles_available=["base"], clusters={},
            trends_available=True, recommendation_engine_ready=True,
        )
        assert r.vacancies_loaded

    def test_profile_short(self):
        r = ProfileShort(profile_name="p1", target_level="junior", skills_count=5)
        assert r.profile_name == "p1"

    def test_profiled_eval(self):
        r = ProfiledEval(market_coverage_score=0.5, error=None)
        assert r.market_coverage_score == 0.5

    def test_profiles_compare(self):
        r = ProfilesCompareResponse(profiles={"p1": ProfiledEval()})
        assert "p1" in r.profiles

    def test_skill_item(self):
        r = SkillItem(skill="python", weight=1.0)
        assert r.skill == "python"

    def test_top_skills(self):
        r = TopSkillsResponse(skills=[SkillItem(skill="python", weight=1.0)])
        assert len(r.skills) == 1

    def test_skill_info(self):
        r = SkillInfoResponse(skill="python", frequency=10, weight=1.0, category="lang", icon="py")
        assert r.skill == "python"

    def test_market_competencies(self):
        r = MarketCompetenciesResponse(skills=[{"s": "python"}], total=1)
        assert r.total == 1

    def test_cluster_summary_item(self):
        r = ClusterSummaryItem(id=1, name="ml", top_skills=["python"])
        assert r.name == "ml"

    def test_level_clusters(self):
        r = LevelClusters(clusters=3, type="kmeans")
        assert r.clusters == 3

    def test_cluster_summary_response(self):
        r = ClusterSummaryResponse(root={"junior": LevelClusters()})
        assert "junior" in r.root

    def test_clusters_by_level(self):
        r = ClustersByLevelResponse(level="junior", clusters=[ClusterSummaryItem(id=1, name="ml", top_skills=["python"])])
        assert r.level == "junior"

    def test_trends_response(self):
        r = TrendsResponse(trends={"rising": [{"s": "python"}]})
        assert "rising" in r.trends

    def test_category_coverage(self):
        r = CategoryCoverage(label="lang", icon="py", total=10, covered=5, percent=50.0)
        assert r.label == "lang"

    def test_taxonomy_coverage(self):
        r = TaxonomyCoverageResponse(coverage={"cat": CategoryCoverage(label="l", icon="i", total=1, covered=1, percent=100)})
        assert "cat" in r.coverage

    def test_profession_item(self):
        r = ProfessionItem(name="dev", domains=[], competency_codes=[], hh_queries=[], aliases=[])
        assert r.name == "dev"

    def test_professions_response(self):
        r = ProfessionsResponse(professions=[ProfessionItem(name="d", domains=[], competency_codes=[], hh_queries=[], aliases=[])], total=1)
        assert r.total == 1

    def test_krm_competency(self):
        r = KRMCompetency(skill_count=2, skills=["python", "sql"])
        assert r.skill_count == 2

    def test_profession_detail(self):
        r = ProfessionDetailResponse(name="dev", domains=[], skill_count=5, skills=[], competency_codes=[], krm_competencies={})
        assert r.name == "dev"

    def test_krm_expertise_item(self):
        r = KRMExpertiseItem(coverage=0.8, total_required=10, covered_skills=[], missing_skills=[])
        assert r.coverage == 0.8

    def test_krm_coverage(self):
        r = KRMCoverageResponse(profession="dev", user_skills=[], competency_coverage={}, avg_coverage=0.5)
        assert r.profession == "dev"

    def test_profession_eval(self):
        r = ProfessionEvalResponse(
            profile="p1", target_profession="dev", target_domains=[],
            profession_coverage=0.7, krm_coverage={}, readiness_score=0.8,
            skill_coverage=0.6, domain_coverage_score=0.5,
        )
        assert r.profile == "p1"

    def test_missing_skill_item(self):
        r = MissingSkillItem(skill="python", frequency=5)
        assert r.skill == "python"

    def test_missing_skills(self):
        r = MissingSkillsResponse(missing_skills=[MissingSkillItem(skill="python", frequency=5)])
        assert len(r.missing_skills) == 1

    def test_dead_skills(self):
        r = DeadSkillsResponse(dead_skills=["cobol"])
        assert "cobol" in r.dead_skills

    def test_pipeline_task_status(self):
        r = PipelineTaskStatus(task_id="t1", status="running", message="working")
        assert r.task_id == "t1"

    def test_gap_progress(self):
        r = GapProgressResponse(pct=50.0, message="half", stage="gap")
        assert r.pct == 50.0

    def test_pipeline_task_list(self):
        r = PipelineTaskListResponse(tasks=[PipelineTaskStatus(task_id="t1", status="done", message="ok")], total=1)
        assert r.total == 1

    def test_pipeline_status(self):
        r = PipelineStatusResponse(
            clusters={}, clusters_all_ready=False, ltr_model=True,
            recommendations={}, recommendations_all_ready=False,
            skill_weights=True, scripts={},
        )
        assert r.ltr_model

    def test_pipeline_simple(self):
        r = PipelineSimpleResponse(status="ok", message="done")
        assert r.status == "ok"

    def test_cache_refresh(self):
        r = CacheRefreshResponse(status="ok", message="done", removed=[], next_step="")
        assert r.status == "ok"

    def test_vacancy_item(self):
        r = VacancyItem(name="Dev", experience="1-3", employer_name="Co", area="Moscow", skills=[])
        assert r.name == "Dev"

    def test_vacancies_response(self):
        r = VacanciesResponse(
            items=[VacancyItem(name="Dev", experience="1-3", employer_name="Co", area="Moscow", skills=[])],
            total=1, limit=10, offset=0, has_more=False,
        )
        assert r.total == 1

    def test_vacancy_detail(self):
        r = VacancyDetailResponse()
        assert r.description == ""

    def test_vacancy_stats(self):
        r = VacancyStatsResponse(total=10, by_experience={"junior": 5}, salary={"avg": 100})
        assert r.total == 10

    def test_regions(self):
        r = RegionsResponse(regions=["Moscow"], total=1)
        assert "Moscow" in r.regions

    def test_vacancies_by_region(self):
        r = VacanciesByRegionResponse(region="Moscow", count=5, limit=10, vacancies=[{"id": 1}])
        assert r.region == "Moscow"
