/**
 * Central documentation configuration.
 *
 * Single source of truth for the whole documentation site:
 * - site links (repo / app)
 * - the complete navigation hierarchy (sections -> pages -> sub-groups)
 * - which sections appear on the homepage cards (homeText / homeCta are the
 *   short card text and link label; summary is the fallback)
 *
 * A page entry becomes live once it has href. Entries without href are
 * planned pages and render as muted "planned" items in navigation.
 * Sidebar, breadcrumbs, previous/next and the homepage cards are all
 * derived from this file - do not edit HTML shell files to change
 * structure.
 */

window.YAPAT_DOCS = {

  repoUrl: "https://github.com/yapat-app/YAPAT",
  appUrl: "https://yapat.ni.dfki.de/",

  sections: [
    {
      id: "getting-started",
      label: "Getting Started",
      summary: "What YAPAT is, how to install it, and your first annotation session.",
      homeText: "Install YAPAT and run a first annotation session, from dataset to exported labels.",
      homeCta: "Start here",
      pages: [
        {
          id: "what-is-yapat",
          title: "What is YAPAT?",
          href: "getting-started/what-is-yapat/"
        },
        {
          id: "installation",
          title: "Installation",
          href: "getting-started/installation/"
        },
        {
          id: "first-workflow",
          title: "First Workflow",
          href: "getting-started/first-workflow/"
        }
      ]
    },
    {
      id: "guides",
      label: "Guides",
      summary: "Step-by-step guides for datasets, the Annotation Hub, label management, and teams.",
      homeText: "Step-by-step help for datasets, the Annotation Hub, label spaces, and teams.",
      homeCta: "Read the guides",
      pages: [
        { id: "datasets", title: "Datasets", href: "guides/datasets/" },
        { id: "annotation-hub", title: "Annotation Hub", href: "guides/annotation-hub/" },
        { id: "label-management", title: "Label Management", href: "guides/label-management/" },
        { id: "teams", title: "Teams", href: "guides/teams/" }
      ]
    },
    {
      id: "concepts",
      label: "Concepts",
      summary: "The ideas behind YAPAT: active learning, filters and model scores, embeddings, WSSED, and label spaces.",
      homeText: "Active learning, embeddings, model scores, sound event detection, and label spaces.",
      homeCta: "Learn the concepts",
      pages: [
        {
          id: "pam-and-annotations",
          title: "PAM and Annotations",
          href: "concepts/pam-and-annotations/"
        },
        {
          id: "active-learning",
          title: "Active Learning",
          href: "concepts/annotation-feeds/active-learning/"
        },
        {
          id: "filter",
          title: "Filters and Model Scores",
          href: "concepts/annotation-feeds/filter/"
        },
        {
          id: "embeddings",
          title: "Embeddings",
          href: "concepts/embeddings/"
        },
        {
          id: "wssed",
          title: "Sound Event Detection (WSSED)",
          href: "concepts/wssed/"
        },
        {
          id: "label-space",
          title: "Label Space",
          pages: [
            { id: "what-is-a-label-space", title: "What is a Label Space?", href: "concepts/label-space/what-is-a-label-space/" },
            { id: "grounding", title: "Grounding", href: "concepts/label-space/grounding/" },
            { id: "managing-versions", title: "Freezing a Label Space", href: "concepts/label-space/managing-versions/" }
          ]
        }
      ]
    },
    {
      id: "workflow",
      label: "Developers",
      summary: "How YAPAT is built, how data moves through it, and how to run, test, and extend the code.",
      homeText: "Architecture, development setup, the programming interface, and extension points.",
      homeCta: "Open developer docs",
      pages: [
        { id: "architecture", title: "Architecture", href: "workflow/architecture/" },
        { id: "workflow-overview", title: "Pipeline Overview", href: "workflow/workflow-overview/" },
        { id: "import-export", title: "Import & Export", href: "workflow/import-export/" },
        { id: "snippets-and-embeddings", title: "Snippets & Embeddings", href: "workflow/snippets-and-embeddings/" },
        { id: "development", title: "Development Setup", href: "workflow/development/" },
        { id: "extending", title: "Extending YAPAT", href: "workflow/extending/" }
      ]
    },
    {
      id: "reference",
      label: "API Reference",
      summary: "The REST API, the active-learning endpoints, and the backend data models.",
      pages: [
        { id: "rest-api", title: "REST API", href: "reference/rest-api/" },
        { id: "feed-methods", title: "Active Learning API", href: "reference/feed-methods/" },
        { id: "data-models", title: "Data Models", href: "reference/data-models/" }
      ]
    },
    {
      id: "operations",
      label: "Operations",
      summary: "Deployment, Docker, environment variables, and troubleshooting.",
      pages: [
        { id: "deployment", title: "Deployment", href: "operations/deployment/" },
        { id: "docker", title: "Docker", href: "operations/docker/" },
        { id: "environment", title: "Environment Setup", href: "operations/environment/" },
        { id: "troubleshooting", title: "Troubleshooting", href: "operations/troubleshooting/" }
      ]
    },
    {
      id: "about",
      label: "About",
      summary: "The project, its license, the publications behind it, and how to contribute.",
      pages: [
        { id: "project", title: "Project", href: "about/project/" },
        { id: "research-context", title: "Publications", href: "about/research-context/" },
        { id: "contributing", title: "Contributing", href: "about/contributing/" }
      ]
    }
  ],

  homeCards: ["getting-started", "guides", "concepts", "workflow"]
};
