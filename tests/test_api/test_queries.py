"""Tests for GraphQL queries — filters, pagination, and data integrity."""

from __future__ import annotations

import pytest


class TestHealthQuery:
    def test_health(self, graphql):
        result = graphql("{ health }")
        assert result["data"]["health"] == "ok"

    def test_health_no_errors(self, graphql):
        result = graphql("{ health }")
        assert "errors" not in result


class TestCohortNIVQuery:
    def test_returns_all(self, graphql):
        result = graphql("{ cohortNiv { totalCount } }")
        assert result["data"]["cohortNiv"]["totalCount"] == 8

    def test_filter_by_cohort_type(self, graphql):
        result = graphql('{ cohortNiv(cohortType: "refugee") { totalCount edges { node { cohortType } } } }')
        assert result["data"]["cohortNiv"]["totalCount"] == 1
        assert result["data"]["cohortNiv"]["edges"][0]["node"]["cohortType"] == "refugee"

    def test_pagination_first(self, graphql):
        result = graphql("{ cohortNiv(first: 3) { totalCount pageInfo { hasNextPage } edges { cursor } } }")
        data = result["data"]["cohortNiv"]
        assert len(data["edges"]) == 3
        assert data["pageInfo"]["hasNextPage"] is True
        assert data["totalCount"] == 8

    def test_pagination_after(self, graphql):
        # Get first page
        page1 = graphql("{ cohortNiv(first: 3) { edges { cursor } pageInfo { endCursor } } }")
        cursor = page1["data"]["cohortNiv"]["edges"][-1]["cursor"]

        # Get second page
        page2 = graphql(f'{{ cohortNiv(first: 3, after: "{cursor}") {{ edges {{ node {{ cohortType }} }} }} }}')
        assert len(page2["data"]["cohortNiv"]["edges"]) == 3


class TestLambdaResultsQuery:
    def test_returns_list(self, graphql):
        result = graphql("{ lambdaResults { variable coefficient } }")
        data = result["data"]["lambdaResults"]
        assert len(data) >= 5
        assert "variable" in data[0]
        assert "coefficient" in data[0]


class TestWelfareLossQuery:
    def test_returns_data(self, graphql):
        result = graphql("{ welfareLoss { totalCount } }")
        assert result["data"]["welfareLoss"]["totalCount"] > 0

    def test_filter_by_province(self, graphql):
        result = graphql('{ welfareLoss(province: "ON") { edges { node { province } } } }')
        for edge in result["data"]["welfareLoss"]["edges"]:
            assert edge["node"]["province"] == "ON"

    def test_filter_by_year_range(self, graphql):
        result = graphql("{ welfareLoss(yearMin: 2020, yearMax: 2022) { edges { node { year } } } }")
        years = [e["node"]["year"] for e in result["data"]["welfareLoss"]["edges"]]
        assert all(2020 <= y <= 2022 for y in years)


class TestDecompositionQuery:
    def test_returns_data(self, graphql):
        result = graphql("{ decomposition { totalCount } }")
        assert result["data"]["decomposition"]["totalCount"] > 0

    def test_filter_by_dimension(self, graphql):
        result = graphql('{ decomposition(dimension: "housing_vacancy") { edges { node { dimension } } } }')
        for edge in result["data"]["decomposition"]["edges"]:
            assert edge["node"]["dimension"] == "housing_vacancy"


class TestCounterfactualQuery:
    def test_returns_data(self, graphql):
        result = graphql("{ counterfactual { totalCount } }")
        assert result["data"]["counterfactual"]["totalCount"] > 0

    def test_filter_by_scenario(self, graphql):
        result = graphql('{ counterfactual(scenario: "aci_equal") { edges { node { scenario } } } }')
        for edge in result["data"]["counterfactual"]["edges"]:
            assert edge["node"]["scenario"] == "aci_equal"

    def test_filter_by_province_and_year(self, graphql):
        result = graphql('{ counterfactual(province: "BC", yearMin: 2022, yearMax: 2023) { edges { node { year province } } } }')
        for edge in result["data"]["counterfactual"]["edges"]:
            assert edge["node"]["province"] == "BC"
            assert 2022 <= edge["node"]["year"] <= 2023


class TestMasterPanelQuery:
    def test_returns_data(self, graphql):
        result = graphql("{ masterPanel { totalCount } }")
        assert result["data"]["masterPanel"]["totalCount"] > 0

    def test_nested_types(self, graphql):
        result = graphql("""
        {
            masterPanel(first: 1) {
                edges {
                    node {
                        year province cohortType
                        housing { startsAnnual vacancyRate }
                        labour { unemploymentRate medianWageHourly }
                        aci { housingHeavy equal fiscalHeavy }
                    }
                }
            }
        }
        """)
        node = result["data"]["masterPanel"]["edges"][0]["node"]
        assert "housing" in node
        assert "labour" in node
        assert "aci" in node

    def test_filter_province_and_cohort(self, graphql):
        result = graphql('{ masterPanel(province: "QC", cohortType: "refugee") { edges { node { province cohortType } } } }')
        for edge in result["data"]["masterPanel"]["edges"]:
            assert edge["node"]["province"] == "QC"
            assert edge["node"]["cohortType"] == "refugee"

    def test_pagination_consistency(self, graphql):
        p1 = graphql("{ masterPanel(province: \"ON\", first: 5) { totalCount edges { cursor } } }")
        total = p1["data"]["masterPanel"]["totalCount"]
        cursor = p1["data"]["masterPanel"]["edges"][-1]["cursor"]
        p2 = graphql(f'{{ masterPanel(province: "ON", first: 5, after: "{cursor}") {{ edges {{ node {{ year }} }} }} }}')
        # Second page should have different data
        if p2["data"]["masterPanel"]["edges"]:
            assert len(p2["data"]["masterPanel"]["edges"]) <= 5


class TestEmptyResults:
    def test_nonexistent_province(self, graphql):
        result = graphql('{ masterPanel(province: "ZZ") { totalCount edges { node { province } } } }')
        assert result["data"]["masterPanel"]["totalCount"] == 0
        assert result["data"]["masterPanel"]["edges"] == []

    def test_nonexistent_year_range(self, graphql):
        result = graphql("{ welfareLoss(yearMin: 2050, yearMax: 2051) { totalCount } }")
        assert result["data"]["welfareLoss"]["totalCount"] == 0
