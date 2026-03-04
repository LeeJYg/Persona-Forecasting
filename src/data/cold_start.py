"""Cold-start 아이템 샘플러 및 데이터 분리기 (config-driven, store-item 기준).

핵심 설계 원칙
--------------
- **sampling_level = "store_item"** (Option B):
    id = item_id × store_id 단위로 샘플링.
    특정 매장(target_store)에서의 특정 상품을 cold-start 시뮬레이션.
- **cross_store_info = false**:
    cold-start 아이템(id)은 '다른 매장에서의 동일 item_id 이력'도 warm_train에서 제외.
  = true:
    target_store만 cold, 나머지 매장의 동일 item_id 이력은 warm_train에 포함.
    → 미래 실험 조건 분기를 위해 파라미터로 제어.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DotDict, load_config

logger = logging.getLogger(__name__)


class ColdStartSampler:
    """
    M5 데이터에서 cold-start 시뮬레이션용 store-item 쌍을 샘플링하고
    warm/cold 분리 데이터셋을 생성한다.

    Args:
        config: load_config()로 반환된 DotDict.
        root: 프로젝트 루트 경로.
    """

    def __init__(
        self,
        config: DotDict | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.root = Path(root) if root else Path(__file__).parent.parent.parent.resolve()

        cs_cfg = self.config.experiment.cold_start
        self.seed: int = self.config.experiment.seed
        self.n_cold_items: int = cs_cfg.n_cold_items
        self.target_store: str = cs_cfg.target_store
        self.category_balance: bool = cs_cfg.category_balance
        self.tier_weights: dict[str, float] = dict(cs_cfg.tier_weights)
        self.cross_store_info: bool = cs_cfg.cross_store_info

        np.random.seed(self.seed)
        logger.info(
            "ColdStartSampler initialized | target_store=%s, n=%d, "
            "cross_store_info=%s, seed=%d",
            self.target_store,
            self.n_cold_items,
            self.cross_store_info,
            self.seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_item_stats(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        target_store 내 각 item_id의 판매량 통계 및 tier를 계산한다.

        Args:
            train: M5 학습 데이터프레임 (전체 매장 포함).

        Returns:
            pd.DataFrame: item_id, cat_id, dept_id, total_sales,
                          mean_sales, std_sales, sales_tier 컬럼 포함.
        """
        logger.info("Computing item stats for store=%s ...", self.target_store)
        store_train = train[train["store_id"] == self.target_store]
        if store_train.empty:
            raise ValueError(
                f"No data found for target_store='{self.target_store}'. "
                "Check config.experiment.cold_start.target_store."
            )

        stats = (
            store_train.groupby("item_id")
            .agg(
                total_sales=("sales", "sum"),
                mean_sales=("sales", "mean"),
                std_sales=("sales", "std"),
                cat_id=("cat_id", "first"),
                dept_id=("dept_id", "first"),
            )
            .reset_index()
        )
        stats["sales_tier"] = pd.qcut(
            stats["total_sales"], q=3, labels=["Low", "Medium", "High"]
        )
        logger.info(
            "Stats computed: %d items in %s | tier dist: %s",
            len(stats),
            self.target_store,
            stats["sales_tier"].value_counts().to_dict(),
        )
        return stats

    def sample_cold_ids(self, item_stats: pd.DataFrame) -> list[str]:
        """
        tier 가중치와 카테고리 균형을 반영해 cold-start id(item_id × store_id) 목록을 샘플링한다.

        Args:
            item_stats: compute_item_stats()의 반환값.

        Returns:
            list[str]: 샘플링된 id 목록 (f"{item_id}_{target_store}" 형식).
        """
        logger.info(
            "Sampling %d cold-start items (balance=%s) ...",
            self.n_cold_items,
            self.category_balance,
        )
        categories = sorted(item_stats["cat_id"].unique())
        n_cats = len(categories)
        base_per_cat = self.n_cold_items // n_cats
        remainder = self.n_cold_items % n_cats

        sampled_item_ids: list[str] = []
        for i, cat in enumerate(categories):
            cat_n = base_per_cat + (1 if i < remainder else 0)
            cat_items = item_stats[item_stats["cat_id"] == cat]

            # tier별 목표 개수 계산 (반올림 오차는 Medium에 흡수)
            tier_targets: dict[str, int] = {
                tier: int(cat_n * w)
                for tier, w in self.tier_weights.items()
            }
            tier_targets["Medium"] += cat_n - sum(tier_targets.values())

            for tier, tier_n in tier_targets.items():
                tier_items = cat_items[cat_items["sales_tier"] == tier]
                available = len(tier_items)
                if available < tier_n:
                    logger.warning(
                        "%s/%s: requested %d but only %d available → using all",
                        cat, tier, tier_n, available,
                    )
                    tier_n = available
                sampled = tier_items.sample(n=tier_n, random_state=self.seed)
                sampled_item_ids.extend(sampled["item_id"].tolist())

            sampled_in_cat = len(
                [x for x in sampled_item_ids
                 if x in cat_items["item_id"].values]
            )
            logger.info("  %s: %d items sampled", cat, sampled_in_cat)

        # id = item_id_store_id 형식으로 변환
        cold_ids = [f"{iid}_{self.target_store}" for iid in sampled_item_ids]
        logger.info("Total cold store-item pairs: %d", len(cold_ids))
        return cold_ids

    def split(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        cold_ids: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        cold_ids를 기준으로 warm/cold 분리 데이터셋을 생성한다.

        cross_store_info 파라미터에 따라 warm_train 구성이 달라진다:
          - False: cold item_ids의 모든 매장 이력을 warm_train에서 제외
          - True:  target_store 이력만 제외 (다른 매장 이력은 warm_train에 포함)

        Args:
            train: M5 전체 학습 데이터프레임.
            test: M5 전체 테스트 데이터프레임.
            cold_ids: sample_cold_ids()의 반환값.

        Returns:
            (warm_train, warm_test, cold_train, cold_test)
        """
        # cold item_id 집합 (store suffix 제거)
        cold_item_id_only = {cid.replace(f"_{self.target_store}", "")
                              for cid in cold_ids}

        # test: target_store의 cold item_id 행만 cold_test로 분리
        # id 컬럼 포맷이 "{item_id}_{store_id}_validation" 등 가변적이므로
        # item_id + store_id 두 컬럼의 조합으로 마스킹
        test_cold_mask = (
            test["item_id"].isin(cold_item_id_only)
            & (test["store_id"] == self.target_store)
        )

        # warm_train: cross_store_info에 따라 분기
        if self.cross_store_info:
            # target_store의 cold item만 제외 → 다른 매장 이력은 유지
            train_cold_mask = (
                train["item_id"].isin(cold_item_id_only)
                & (train["store_id"] == self.target_store)
            )
            logger.info(
                "cross_store_info=True: excluding only %s records "
                "for %d cold item_ids from train",
                self.target_store,
                len(cold_item_id_only),
            )
        else:
            # cold item_ids의 모든 매장 이력 제외
            train_cold_mask = train["item_id"].isin(cold_item_id_only)
            logger.info(
                "cross_store_info=False: excluding all store histories "
                "for %d cold item_ids from train",
                len(cold_item_id_only),
            )

        warm_train = train[~train_cold_mask].copy()
        warm_test = test[~test_cold_mask].copy()
        cold_train = train.iloc[0:0].copy()  # 빈 DataFrame (스키마 유지)
        cold_test = test[test_cold_mask].copy()

        logger.info(
            "Split result | warm_train=%s, warm_test=%s, "
            "cold_train=%s (empty), cold_test=%s",
            f"{len(warm_train):,}",
            f"{len(warm_test):,}",
            f"{len(cold_train):,}",
            f"{len(cold_test):,}",
        )
        actual_cold_items = cold_test["id"].nunique()
        if actual_cold_items != len(cold_ids):
            logger.warning(
                "Expected %d cold ids in test, found %d. "
                "Some ids may not appear in the test period.",
                len(cold_ids),
                actual_cold_items,
            )
        return warm_train, warm_test, cold_train, cold_test

    def save(
        self,
        warm_train: pd.DataFrame,
        warm_test: pd.DataFrame,
        cold_train: pd.DataFrame,
        cold_test: pd.DataFrame,
        cold_ids: list[str],
        item_stats: pd.DataFrame,
    ) -> Path:
        """
        분리 데이터셋과 메타데이터를 config.paths.cold_start_dir에 저장한다.

        Args:
            warm_train, warm_test, cold_train, cold_test: split()의 반환값.
            cold_ids: 샘플링된 cold store-item id 목록.
            item_stats: compute_item_stats()의 반환값.

        Returns:
            Path: 출력 디렉토리 경로.
        """
        out_dir = self.root / self.config.paths.cold_start_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving to %s ...", out_dir)

        warm_train.to_csv(out_dir / "warm_train.csv", index=False)
        warm_test.to_csv(out_dir / "warm_test.csv", index=False)
        cold_train.to_csv(out_dir / "cold_train.csv", index=False)
        cold_test.to_csv(out_dir / "cold_test.csv", index=False)

        # cold id 목록
        pd.DataFrame({"id": cold_ids}).to_csv(
            out_dir / "cold_ids.csv", index=False
        )

        # cold 아이템 통계 (item_id + store_id 부착)
        cold_item_id_only = [cid.replace(f"_{self.target_store}", "")
                              for cid in cold_ids]
        cold_stats = item_stats[item_stats["item_id"].isin(cold_item_id_only)].copy()
        cold_stats.to_csv(out_dir / "cold_item_stats.csv", index=False)

        # 메타데이터
        metadata: dict[str, Any] = {
            "n_cold_ids": len(cold_ids),
            "target_store": self.target_store,
            "sampling_level": self.config.experiment.cold_start.sampling_level,
            "cross_store_info": self.cross_store_info,
            "seed": self.seed,
            "category_distribution": (
                cold_stats["cat_id"].value_counts().to_dict()
            ),
            "tier_distribution": (
                cold_stats["sales_tier"].astype(str).value_counts().to_dict()
            ),
            "warm_train_rows": len(warm_train),
            "warm_test_rows": len(warm_test),
            "cold_train_rows": len(cold_train),
            "cold_test_rows": len(cold_test),
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved: %s", [p.name for p in sorted(out_dir.iterdir())])
        return out_dir

    def summary(self, cold_ids: list[str], item_stats: pd.DataFrame) -> None:
        """샘플링 결과 요약을 로그로 출력한다."""
        cold_item_id_only = [cid.replace(f"_{self.target_store}", "")
                              for cid in cold_ids]
        cs = item_stats[item_stats["item_id"].isin(cold_item_id_only)]
        logger.info("=" * 60)
        logger.info("COLD-START SAMPLING SUMMARY")
        logger.info("=" * 60)
        logger.info("target_store      : %s", self.target_store)
        logger.info("n_cold_ids        : %d", len(cold_ids))
        logger.info("cross_store_info  : %s", self.cross_store_info)
        logger.info("Category dist     : %s", cs["cat_id"].value_counts().to_dict())
        logger.info(
            "Tier dist         : %s",
            cs["sales_tier"].astype(str).value_counts().to_dict(),
        )
        logger.info(
            "Mean sales (cold) : mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
            cs["mean_sales"].mean(),
            cs["mean_sales"].std(),
            cs["mean_sales"].min(),
            cs["mean_sales"].max(),
        )
        logger.info("=" * 60)
