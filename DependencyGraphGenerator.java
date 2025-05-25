import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class DependencyGraphGenerator {
    // Graph structure to hold nodes (classes/methods) and edges (dependencies/calls)
    static class Graph {
        List<Node> nodes = new ArrayList<>();
        List<Edge> edges = new ArrayList<>();
    }

    static class Node {
        String id; // e.g., "repo1:com.example.UserController"
        String type; // e.g., "Class", "Method"
        String repo; // Repository name
        String label; // Display name (e.g., class or method name)
        Map<String, String> attributes; // Additional metadata (e.g., annotations)

        Node(String id, String type, String repo, String label) {
            this.id = id;
            this.type = type;
            this.repo = repo;
            this.label = label;
            this.attributes = new HashMap<>();
        }
    }

    static class Edge {
        String source; // Source node ID
        String target; // Target node ID
        String type; // e.g., "Dependency", "MethodCall"

        Edge(String source, String target, String type) {
            this.source = source;
            this.target = target;
            this.type = type;
        }
    }

    private final Graph graph = new Graph();
    private final Set<String> nodeIds = new HashSet<>(); // Track unique nodes

    // Main method to process repositories and generate JSON
    public static void main(String[] args) {
        // Example repository paths (replace with actual paths)
        String[] repoPaths = {
            "/repos/repo1",
            "/repos/repo2",
            "/repos/repo3"
        };
        String outputJsonPath = "dependency_graph.json";

        DependencyGraphGenerator generator = new DependencyGraphGenerator();
        for (String repoPath : repoPaths) {
            String repoName = new File(repoPath).getName();
            generator.parseRepository(repoPath, repoName);
        }
        generator.writeGraphToJson(outputJsonPath);
    }

    // Parse all Java files in a repository
    private void parseRepository(String repoPath, String repoName) {
        try (Stream<Path> paths = Files.walk(Paths.get(repoPath))) {
            paths.filter(path -> path.toString().endsWith(".java"))
                 .forEach(path -> parseJavaFile(path, repoName));
        } catch (IOException e) {
            System.err.println("Error walking repository " + repoName + ": " + e.getMessage());
        }
    }

    // Parse a single Java file
    private void parseJavaFile(Path filePath, String repoName) {
        try {
            CompilationUnit cu = StaticJavaParser.parse(filePath);
            new ClassVisitor(repoName).visit(cu, null);
        } catch (IOException e) {
            System.err.println("Error parsing file " + filePath + ": " + e.getMessage());
        }
    }

    // Visitor to extract classes, methods, and dependencies
    private class ClassVisitor extends VoidVisitorAdapter<Void> {
        private final String repoName;

        ClassVisitor(String repoName) {
            this.repoName = repoName;
        }

        @Override
        public void visit(ClassOrInterfaceDeclaration clazz, Void arg) {
            String packageName = clazz.getFullyQualifiedName().orElse("default");
            String classId = repoName + ":" + packageName;

            // Add class node
            if (!nodeIds.contains(classId)) {
                Node classNode = new Node(classId, "Class", repoName, clazz.getNameAsString());
                clazz.getAnnotations().forEach(a -> classNode.attributes.put("annotation", a.getNameAsString()));
                graph.nodes.add(classNode);
                nodeIds.add(classId);
            }

            // Process fields for dependencies (e.g., @Autowired)
            for (FieldDeclaration field : clazz.getFields()) {
                field.getAnnotations().forEach(a -> {
                    if (a.getNameAsString().equals("Autowired")) {
                        String targetType = field.getVariable(0).getTypeAsString();
                        String targetId = findTargetClassId(targetType, repoName);
                        if (targetId != null) {
                            graph.edges.add(new Edge(classId, targetId, "Dependency"));
                        }
                    }
                });
            }

            // Process methods and method calls
            for (MethodDeclaration method : clazz.getMethods()) {
                String methodId = classId + "#" + method.getSignature().asString();
                Node methodNode = new Node(methodId, "Method", repoName, method.getNameAsString());
                method.getAnnotations().forEach(a -> methodNode.attributes.put("annotation", a.getNameAsString()));
                if (!nodeIds.contains(methodId)) {
                    graph.nodes.add(methodNode);
                    nodeIds.add(methodId);
                }

                // Find method calls
                method.findAll(MethodCallExpr.class).forEach(call -> {
                    String targetMethod = call.getScope().map(s -> s.toString() + "." + call.getNameAsString()).orElse(call.getNameAsString());
                    String targetId = resolveMethodCallTarget(targetMethod, repoName);
                    if (targetId != null) {
                        graph.edges.add(new Edge(methodId, targetId, "MethodCall"));
                    }
                });
            }

            super.visit(clazz, arg);
        }
    }

    // Resolve target class ID (simplified; enhance with package resolution)
    private String findTargetClassId(String typeName, String repoName) {
        // Search all repos for the class (simplified; assumes unique class names)
        for (String repo : Arrays.asList("repo1", "repo2", "repo3")) {
            String potentialId = repo + ":com.example." + typeName; // Adjust package as needed
            if (nodeIds.contains(potentialId)) {
                return potentialId;
            }
        }
        return null;
    }

    // Resolve method call target (simplified; enhance with symbol resolution)
    private String resolveMethodCallTarget(String methodCall, String repoName) {
        // Placeholder: Map method call to target class/method (e.g., via symbol table or annotations)
        // For Spring, check if method is in a @RestController or @Service
        return null; // Implement based on codebase specifics
    }

    // Write graph to JSON file
    private void writeGraphToJson(String outputPath) {
        try (FileWriter writer = new FileWriter(outputPath)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(graph, writer);
        } catch (IOException e) {
            System.err.println("Error writing JSON: " + e.getMessage());
        }
    }
}