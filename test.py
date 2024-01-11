import sqlparse
import html

str = 'SELECT\n    HLL_COUNT.MERGE(id_sketch) AS user_count\nFROM\n    cdp-demo-flocquet.publisher_1_dataset.hll_user_aggregates,\n    (\n    SELECT\n        brand_name\n    FROM\n        cdp-demo-flocquet.publisher_1_dataset.products_aggregates\n    WHERE\n        DATE(session_day) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)\n    GROUP BY\n        brand_name\n    ORDER BY\n        SUM(daily_product_purchased_items) DESC\n    LIMIT 1\n    ) AS top_brand\nWHERE\n    EXISTS (\n    SELECT\n        1\n    FROM\n        UNNEST(daily_purchased_products_by_brands) AS daily_purchased_products_by_brands_unnested\n    WHERE\n        daily_purchased_products_by_brands_unnested.brand_name = top_brand.brand_name\n    )'
str = sqlparse.format(str, reindent=True, indent_width=4, strip_comments=True, )
result = {'result': html.escape(str)}
print(result)