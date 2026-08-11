# Agent 行为准则

## 核心原则

在执行任何任务之前，必须先查阅 Memory_Bank 中的文件：

1. **先读 Progress.md** — 了解当前执行到了哪一步，上一步的状态
2. **再读 Architecture.md** — 了解项目的结构、模块边界、接口标准
3. **按照 Implementation_Plan.md 执行** — 按步骤顺序执行，明确每步的验收标准

## 执行规则

- 每完成一个步骤，必须更新 Progress.md 中对应步骤的状态
- 如果发现 Architecture.md 与实际代码不符，必须同步更新 Architecture.md
- 如果发现无法执行的问题（依赖缺失、接口不一致），**立即中断并提问**，不要擅自绕过
- 所有改动完成后，确认 Memory_Bank 中的文件与实际工程状态一致

## 禁止事项

- 不要跳过任何步骤直接"直觉式"编码
- 不要在未经确认的情况下修改 Architecture.md 中定义的接口
- 不要忽略 Progress.md 中标记为 `blocked` 的步骤

## 文件索引

| 文件 | 位置 | 用途 |
|------|------|------|
| Implementation_Plan.md | Memory_Bank/ | 实施计划与验收标准 |
| Architecture.md | Memory_Bank/ | 项目结构与接口标准 |
| Progress.md | Memory_Bank/ | 执行进度状态 |
| Agent.md | ./ | 本文件 |