# 电商短视频通用模板实现说明

## 范围

当前实现提供 `ecommerce_ugc_v1` 通用模板，并内置两个数据预设：

- `food_v1`：食品、农产品、玉米等商品。
- `beauty_v1`：美妆、礼盒、护肤和品牌商品。

模板负责把已确认的商品事实、镜头角色和广告图层编译成确定性的 MPT 任务配置。新增品类只需要提供新的预设 JSON，不需要复制渲染代码。

参考视频只用于确定视觉结构。参考文件中的烧录字幕、价格和促销事实不能直接作为生成素材或业务输入。

## 执行链路

```text
creative_plan.json
  -> scripts/creative_plan.py
  -> MPT VideoParams（模板模式关闭原生字幕）
  -> scripts/post_process_v2.py
  -> 文本/图片/水印图层
  -> scripts/output_profiles.py
  -> ffprobe 输出校验
```

旧版没有 `creative_plan` 的任务仍使用 v1 后处理和原有字幕流程；两个版本通过 `scripts/post_process_dispatch.py` 按 `schema_version` 显式分流。

## 创意计划

```json
{
  "schema_version": 1,
  "template_id": "ecommerce_ugc_v1",
  "category_preset": "food_v1",
  "facts": {
    "product_name": "东北糯玉米",
    "price_text": "¥3.9起",
    "quantity_text": "10根",
    "cta_text": "点击领取优惠",
    "disclaimer_text": "价格及优惠以实际页面为准"
  },
  "scenes": [
    {
      "role": "hook",
      "narration": "姐妹们快看",
      "visual_query": "成品玉米整齐堆放",
      "duration_seconds": 2.0
    },
    {
      "role": "feature",
      "narration": "皮薄粒满",
      "visual_query": "玉米开箱特写",
      "duration_seconds": 2.0
    },
    {
      "role": "proof",
      "narration": "软糯香甜",
      "visual_query": "掰开玉米展示颗粒",
      "duration_seconds": 2.0
    },
    {
      "role": "offer",
      "narration": "现在下单有优惠",
      "visual_query": "优惠页面",
      "duration_seconds": 2.0
    },
    {
      "role": "cta",
      "narration": "点击领取优惠",
      "visual_query": "购买引导",
      "duration_seconds": 2.0
    }
  ],
  "image_layers": [],
  "output_profile": "portrait-720x1280"
}
```

`scenes` 会被编译成连续的目标时间范围，并自动生成带句号边界的 `video_script`。每个场景可以额外指定 `asset_id` 和源视频起止时间；如果所有场景都提供这三项，selector 会按顺序复制到 `storage/local_videos`，生成可执行的 `local_storyboard_plan`，任务运行时再校验哈希并建立任务级快照。显式场景时长会保留相对比例，并在真实音频时长上重新对齐。

`image_layers` 用于购物车截图、手机页面、优惠卡片等图片素材，字段为：

```text
path, start, end, x_ratio, y_ratio,
width_ratio, height_ratio, fit_mode, opacity
```

路径在后处理阶段必须位于受管 `resource/` 目录；图层越界、文件缺失、超大图片或时间非法都会明确失败。文字层按目标宽度自动换行，单层文本长度和图片文件/像素数量也有上限。

## Selector 用法

```bash
python -m scripts.selector \
  --request request.json \
  --manifest manifest/assets.json \
  --output-root output \
  --mpt-root . \
  --category-presets presets.json
```

任务中加入 `creative_plan` 后，selector 会：

1. 校验商品事实和场景角色。
2. 生成 v2 `post_process_specs/{task_index}.json`。
3. 将规范化计划保存到 `creative_plans/{task_index}.json`。
4. 在 MPT batch 中显式关闭原生字幕，避免后处理字幕重复。

## 输出规格

内置模板 profile：

- `portrait-720x1280`：匹配低分辨率竖屏食品样本。
- `portrait-1080x1920-high`：匹配高分辨率美妆样本。
- 原有 `portrait-1080x1920`、横屏和方形 profile 保持不变。

输出尺寸、帧率、编码器、像素格式、音频和 faststart 均从 profile 展开后写入任务快照。

## 当前边界

- 参考视频的精确口播时间码尚未自动转写；示例计划中的时长需要使用实际 TTS/SRT 时间重新校准。
- 生产素材必须是无烧录字幕、无重复水印的干净素材。
- 当前 v2 文字动画先支持静态层；复杂动画仍使用旧版 `custom_texts` 或后续扩展。
- 模板模式下图片图层统一位于文字层下方，文字和水印的顺序是固定约定，不提供任意 z-index。
- 商品价格、折扣、功效和时限只能来自已确认事实，LLM 不负责补造。
- 当前 WebUI 提供结构化模板和图片图层 JSON 入口，尚未提供可视化拖拽时间轴。
