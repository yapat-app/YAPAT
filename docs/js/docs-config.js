/**
 * Central documentation configuration.
 *
 * Single source of truth for the whole documentation site:
 * - site links (repo / app)
 * - the complete navigation hierarchy (sections -> pages -> sub-groups)
 * - which sections appear on the homepage cards
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
      summary: "What YAPAT is, what it does, and how to run your first annotation session.",
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
      id: "concepts",
      label: "Concepts",
      summary: "Label spaces, embeddings, active learning, and the Annotation Hub's filters in the user-study version of YAPAT.",
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
      label: "Workflow",
      summary: "The end-to-end pipeline from dataset import to labelled, export-ready data.",
      pages: [
        { id: "workflow-overview", title: "Overview", href: "workflow/workflow-overview/" },
        { id: "import-export", title: "Import & Export", href: "workflow/import-export/" },
        { id: "snippets-and-embeddings", title: "Snippets & Embeddings", href: "workflow/snippets-and-embeddings/" }
      ]
    },
    {
      id: "guides",
      label: "Guides",
      summary: "Step-by-step guides for datasets and WSSED training, the Annotation Hub and study phases, label management, and teams.",
      pages: [
        { id: "datasets", title: "Datasets", href: "guides/datasets/" },
        { id: "annotation-hub", title: "Annotation Hub", href: "guides/annotation-hub/" },
        { id: "label-management", title: "Label Management", href: "guides/label-management/" },
        { id: "teams", title: "Teams", href: "guides/teams/" }
      ]
    },
    {
      id: "reference",
      label: "API Reference",
      summary: "The REST endpoints the user-study interface uses, the active-learning API, and the backend data models.",
      pages: [
        { id: "rest-api", title: "REST API", href: "reference/rest-api/" },
        { id: "feed-methods", title: "Active Learning API", href: "reference/feed-methods/" },
        { id: "data-models", title: "Data Models", href: "reference/data-models/" }
      ]
    },
    {
      id: "operations",
      label: "Operations",
      summary: "Deployment, Docker configuration, and environment variables for the backend and frontend.",
      pages: [
        { id: "deployment", title: "Deployment", href: "operations/deployment/" },
        { id: "docker", title: "Docker", href: "operations/docker/" },
        { id: "environment", title: "Environment Setup", href: "operations/environment/" }
      ]
    },
    {
      id: "about",
      label: "About",
      summary: "Project background, repository layout, and the research context behind YAPAT.",
      pages: [
        { id: "project", title: "Project", href: "about/project/" },
        { id: "research-context", title: "Research Context", href: "about/research-context/" }
      ]
    }
  ],

  homeCards: ["getting-started", "concepts", "guides", "reference"]
};
