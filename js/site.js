// ============================================================
// 站点文案与学习日志 —— 手动维护的部分
// ============================================================

const SITE = {
  author: "Shijie Zhou",
  title: "Full-Stack Robotics",
  // 扉页
  kicker: "Field notes · Shijie Zhou",
  headline: "Becoming a full‑stack roboticist.",
  lead: "一份写给自己的机器人学习路线：从电机和关节，到强化学习、运动控制、灵巧操作，再到 VLA 和世界模型。边学边写，这个网站和我一起长大。",
  statement: "What I cannot create, I do not understand.",
  statementBy: "Richard Feynman",
  // 个人主页上的那个问题
  question: "How can we build a scalable robot-learning flywheel that unifies perception, whole-body control and dexterous manipulation?",
  home: "https://jerryplain.github.io/",
  repo: "https://github.com/JerryPlain/fullstack_robotics",
  footerLinks: [
    { label: "Home", url: "https://jerryplain.github.io/" },
    { label: "Photography", url: "https://jerryplain.github.io/photography/" },
    { label: "GitHub", url: "https://github.com/JerryPlain/fullstack_robotics" },
    { label: "Email", url: "mailto:jerryplain@outlook.com" },
  ],
  motto: "Life is a long process of mean reversion.",
  copyright: "© 2026 Shijie Zhou",
};

// ------------------------------------------------------------
// 学习日志：最新的写在最上面。body 支持 Markdown 和公式。
// ------------------------------------------------------------
const LOG = [
  {
    date: "2026-11-15",
    title: "Day 0 · 开站",
    tags: ["meta"],
    body: `
搭好了这个网站的骨架：15 条路线、150 节课的目录、170 多篇论文，先写好了 9 节课。目录按 2026 年 11 月的前沿调研补齐了导航、数据、具身智能体和安全评测，所有 arXiv 链接都核对过标题。

规则只有三条：

1. **每学完一课，就点「学完了」**，让地图慢慢亮起来。
2. **每一课末尾的「我的笔记」留给自己**：写下没想通的地方和跑过的实验。
3. **能动手就动手**——What I cannot create, I do not understand.

下一步：把 RL 和 Locomotion 两条线补完，然后在仿真里从零训一个四足。
`,
  },
];
